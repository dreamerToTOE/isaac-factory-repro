"""CPU-only JSONL bridge to the unchanged, allowlisted G4-I1 frozen actor.

The client imports only the standard library, so MuJoCo need not share a
Python installation with Torch/RL-Games.  The server owns recurrent state and
RNG state; a reset is explicit, and transport/model failures are never retried.
Importing this module neither loads the checkpoint nor performs inference.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import selectors
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SHA256 = "f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f"
PROTOCOL = "g4-s2a-frozen-policy-jsonl-v1"
STATE_SHAPE = (2, 1, 1024)
MAX_LINE_BYTES = 4 * 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_vector(value, length, name, *, fp32=False):
    """Validate one unbatched vector; optionally register FP32 transfer bytes."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must be a flat vector of length {length}")
    result = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, numbers.Real):
            raise ValueError(f"{name} must contain real numbers, not booleans")
        component = float(component)
        if not math.isfinite(component):
            raise ValueError(f"{name} contains NaN/Inf")
        if fp32:
            try:
                component = struct.unpack("<f", struct.pack("<f", component))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError(f"{name} is outside finite FP32 range") from exc
            if not math.isfinite(component):
                raise ValueError(f"{name} is outside finite FP32 range")
        result.append(component)
    return result


def _finite_state(value, name, *, require_zero=False):
    if not isinstance(value, list) or len(value) != STATE_SHAPE[0]:
        raise ValueError(f"{name} must have shape {STATE_SHAPE}")
    for layer in value:
        if not isinstance(layer, list) or len(layer) != STATE_SHAPE[1]:
            raise ValueError(f"{name} must have shape {STATE_SHAPE}")
        vector = finite_vector(layer[0], STATE_SHAPE[2], name)
        if require_zero and any(component != 0.0 for component in vector):
            raise ValueError(f"{name} reset is not zero")
    return value


class BridgeError(RuntimeError):
    """Fail-closed bridge error, retaining the actual actor process status."""

    def __init__(self, message, *, returncode=None):
        super().__init__(message)
        self.returncode = returncode


class InferenceClient:
    """Single actor subprocess; responses are finite, shape-checked plain dicts.

    ``reset(seed)`` is mandatory before ``infer(observation19)``.  Returned
    observations/actions are unbatched lists; h/c retain shape [2,1,1024].
    ``latency_seconds`` measures only the server's unchanged actor inference;
    ``bridge_roundtrip_seconds`` additionally includes serialization and IPC.
    """

    def __init__(self, python_executable, checkpoint, *, repo=ROOT,
                 stderr_path=None, timeout_s=60.0):
        if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.timeout_s = float(timeout_s)
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self._sequence = 0
        self._buffer = bytearray()
        self._closed = False
        self._healthy = True
        self._reset = False
        self._infer_index = 0
        self._reset_index = 0
        self._receipt = None
        self._stderr_file = None
        self.process = None
        self._selector = selectors.DefaultSelector()
        try:
            if stderr_path is not None:
                self._stderr_file = Path(stderr_path).open("xb")
            self.process = subprocess.Popen(
                [str(python_executable), "-u", "-m", "scripts.g4s2a_inference",
                 "--serve", "--checkpoint", str(self.checkpoint), "--repo", str(self.repo)],
                cwd=str(self.repo), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr_file, bufsize=0,
            )
            self._selector.register(self.process.stdout, selectors.EVENT_READ)
            ready = self._read_response(0)
            if ready.get("type") != "ready" or ready.get("protocol") != PROTOCOL:
                raise ValueError("actor did not provide the expected ready handshake")
            self._validate_info(ready["info"])
            self.ready = ready
        except Exception as exc:
            self._abort()
            if isinstance(exc, BridgeError):
                raise
            raise BridgeError(f"actor startup failed: {exc}", returncode=self.returncode) from exc

    @property
    def returncode(self):
        return None if self.process is None else self.process.poll()

    def _abort(self):
        self._healthy = False
        self._reset = False
        if self.process is not None and self.process.poll() is None:
            # Only the child this bridge created is terminated on bridge faults.
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._release()

    def _release(self):
        self._selector.close()
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        if self._stderr_file is not None and not self._stderr_file.closed:
            self._stderr_file.close()
        self._closed = True

    def _read_response(self, expected_id):
        deadline = time.monotonic() + self.timeout_s
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                raise BridgeError("actor response timed out", returncode=self.returncode)
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                # wait() obtains an exact exit status rather than a stale poll.
                try:
                    code = self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    code = self.process.poll()
                raise BridgeError("actor stdout closed before a complete response", returncode=code)
            self._buffer.extend(chunk)
            if len(self._buffer) > MAX_LINE_BYTES:
                raise BridgeError("actor response exceeded protocol bound", returncode=self.returncode)
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("id") != expected_id:
            raise ValueError("actor response request ID mismatch")
        if response.get("ok") is not True:
            # The server exits after reporting any failed request; never reuse
            # a possibly advanced recurrent or RNG state.
            try:
                code = self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                code = self.process.poll()
            raise BridgeError(f"actor rejected request: {response.get('error')}", returncode=code)
        return response

    def _request(self, operation, **payload):
        if self._closed or not self._healthy:
            raise BridgeError("actor bridge is closed or unhealthy", returncode=self.returncode)
        self._sequence += 1
        started = time.perf_counter()
        try:
            message = {"id": self._sequence, "op": operation, **payload}
            self.process.stdin.write((json.dumps(message, allow_nan=False) + "\n").encode("utf-8"))
            self.process.stdin.flush()
            response = self._read_response(self._sequence)
            response["bridge_roundtrip_seconds"] = time.perf_counter() - started
            return response
        except Exception as exc:
            self._abort()
            raise BridgeError(f"actor {operation} failed: {exc}", returncode=self.returncode) from exc

    @staticmethod
    def _validate_info(info):
        if (not isinstance(info, dict)
                or info.get("checkpoint_sha256") != PRIMARY_SHA256
                or info.get("checkpoint_sha256_at_load") != PRIMARY_SHA256
                or info.get("device") != "cpu"
                or info.get("state_shape") != list(STATE_SHAPE)
                or info.get("rms_unchanged") is not True
                or info.get("eval_mode") is not True
                or info.get("all_parameters_frozen") is not True):
            raise ValueError("actor identity/frozen-state verification failed")

    def info(self):
        response = self._request("info")
        try:
            self._validate_info(response["info"])
        except Exception as exc:
            self._abort()
            raise BridgeError(str(exc), returncode=self.returncode) from exc
        return response["info"]

    def reset(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        response = self._request("reset", seed=seed)
        try:
            _finite_state(response["h"], "h", require_zero=True)
            _finite_state(response["c"], "c", require_zero=True)
            if (response.get("seed") != seed or response.get("rms_unchanged") is not True
                    or response.get("infer_index") != 0
                    or response.get("reset_index") != self._reset_index + 1):
                raise ValueError("actor reset receipt mismatch")
        except Exception as exc:
            self._abort()
            raise BridgeError(str(exc), returncode=self.returncode) from exc
        self._reset_index += 1
        self._infer_index = 0
        self._reset = True
        return response

    def infer(self, observation):
        if not self._reset:
            raise BridgeError("explicit actor reset is required before inference", returncode=self.returncode)
        observation = finite_vector(observation, 19, "observation", fp32=True)
        response = self._request("infer", observation=observation)
        try:
            for key, size in (("observation", 19), ("normalized", 19),
                              ("mu", 6), ("logstd", 6), ("clipped", 6)):
                finite_vector(response[key], size, key)
            _finite_state(response["h"], "h")
            _finite_state(response["c"], "c")
            if response["observation"] != observation:
                raise ValueError("FP32 observation transfer changed")
            if any(abs(x) > 5.0 for x in response["normalized"]):
                raise ValueError("normalized observation exceeds frozen clipping bound")
            if response["clipped"] != [max(-1.0, min(1.0, x)) for x in response["mu"]]:
                raise ValueError("actor clipped action differs from frozen mean clipping")
            if (response.get("rms_unchanged") is not True
                    or response.get("consumed_discarded_sample") is not True
                    or response.get("infer_index") != self._infer_index + 1
                    or response.get("reset_index") != self._reset_index):
                raise ValueError("actor inference receipt mismatch")
            latency = response["latency_seconds"]
            if isinstance(latency, bool) or not math.isfinite(latency) or latency < 0:
                raise ValueError("invalid actor inference latency")
        except Exception as exc:
            self._abort()
            raise BridgeError(str(exc), returncode=self.returncode) from exc
        self._infer_index += 1
        return response

    def close(self):
        if self._receipt is not None:
            return dict(self._receipt)
        response = None
        if not self._closed and self._healthy:
            response = self._request("close")
            try:
                code = self.process.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired as exc:
                self._abort()
                raise BridgeError("actor did not exit after close", returncode=self.returncode) from exc
            if code != 0:
                self._release()
                raise BridgeError("actor exited unsuccessfully after close", returncode=code)
            self._release()
        self._receipt = {"child_returncode": self.returncode,
                         "healthy": self._healthy, "response": response}
        return dict(self._receipt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self._abort()
        else:
            self.close()
        return False


def _server_info(policy, checkpoint, loaded_sha, infer_count, reset_count, seed, torch):
    digest = file_sha256(checkpoint)
    if digest != loaded_sha or digest != PRIMARY_SHA256:
        raise RuntimeError("checkpoint identity changed during bridge lifetime")
    if not policy.rms_unchanged():
        raise RuntimeError("actor RMS changed")
    return {"checkpoint_sha256": digest, "checkpoint_sha256_at_load": loaded_sha,
            "metadata": policy.metadata, "python_executable": sys.executable,
            "torch_version": str(torch.__version__), "device": str(policy.device),
            "state_shape": list(STATE_SHAPE), "rms_unchanged": True,
            "eval_mode": not policy.model.training and not policy.model.running_mean_std.training,
            "all_parameters_frozen": all(not p.requires_grad for p in policy.model.parameters()),
            "normalization": "unchanged FrozenPolicy.infer exactly once",
            "action": "deterministic mu, clipped to [-1,1]",
            "rng": "torch.manual_seed(reset seed); original discarded Normal.sample consumed",
            "infer_count": infer_count, "reset_count": reset_count, "current_seed": seed}


def _emit(response):
    sys.stdout.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def serve(checkpoint, repo=ROOT):
    """Serve one trusted checkpoint on stdin/stdout; errors exit nonzero."""
    current_id = 0
    try:
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        # RL-Games builders print architecture diagnostics. Keep stdout reserved
        # for JSON protocol even while importing/building the frozen actor.
        with redirect_stdout(sys.stderr):
            import torch
            from scripts.g4i1_policy import FrozenPolicy
            loaded_sha = file_sha256(checkpoint)
            if loaded_sha != PRIMARY_SHA256:
                raise ValueError("checkpoint SHA is not the allowlisted G2 epoch-200 identity")
            policy = FrozenPolicy.from_checkpoint(checkpoint, repo=repo,
                                                  role="g2_epoch200_primary", device="cpu")
        state, seed = None, None
        infer_count = reset_count = infer_index = 0

        def info():
            return _server_info(policy, checkpoint, loaded_sha, infer_count,
                                reset_count, seed, torch)

        _emit({"id": 0, "ok": True, "type": "ready", "protocol": PROTOCOL, "info": info()})
        while True:
            line = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
            if not line:
                raise RuntimeError("parent closed stdin without explicit close")
            if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                raise ValueError("request exceeds protocol line bound")
            request = json.loads(line)
            if not isinstance(request, dict) or type(request.get("id")) is not int:
                raise ValueError("request must have an integer id")
            if request["id"] != current_id + 1:
                raise ValueError("request IDs must be consecutive")
            current_id = request["id"]
            operation = request.get("op")
            keys = {"id", "op"}
            if operation == "reset":
                keys.add("seed")
            elif operation == "infer":
                keys.add("observation")
            if set(request) != keys:
                raise ValueError("request has unexpected or missing fields")
            if operation == "info":
                result = {"info": info()}
            elif operation == "reset":
                seed = request["seed"]
                if type(seed) is not int or not 0 <= seed < 2**63:
                    raise ValueError("seed must be an integer in [0,2**63)")
                torch.manual_seed(seed)
                state = policy.initial_state(1)
                if any(torch.count_nonzero(x).item() for x in state):
                    raise RuntimeError("frozen actor initial state is not zero")
                infer_index = 0
                reset_count += 1
                result = {"seed": seed, "h": state[0].tolist(), "c": state[1].tolist(),
                          "rms_unchanged": policy.rms_unchanged(),
                          "infer_index": infer_index, "reset_index": reset_count}
            elif operation == "infer":
                if state is None:
                    raise RuntimeError("explicit reset is required before inference")
                observation = finite_vector(request["observation"], 19, "observation", fp32=True)
                tensor = torch.tensor([observation], dtype=torch.float32, device="cpu").contiguous()
                started = time.perf_counter()
                with redirect_stdout(sys.stderr):
                    step = policy.infer(tensor, state, consume_rng=True)
                latency = time.perf_counter() - started
                state = step.state
                infer_count += 1
                infer_index += 1
                result = {"observation": step.observation[0].tolist(),
                          "normalized": step.normalized[0].tolist(),
                          "mu": step.mu[0].tolist(), "logstd": step.logstd[0].tolist(),
                          "clipped": step.clipped_action[0].tolist(),
                          "h": step.h.tolist(), "c": step.c.tolist(),
                          "latency_seconds": latency, "rms_unchanged": policy.rms_unchanged(),
                          "consumed_discarded_sample": step.consumed_discarded_sample,
                          "infer_index": infer_index, "reset_index": reset_count,
                          "infer_count": infer_count}
            elif operation == "close":
                _emit({"id": current_id, "ok": True, "info": info(), "closed": True})
                return 0
            else:
                raise ValueError("unsupported actor operation")
            _emit({"id": current_id, "ok": True, **result})
    except Exception as exc:
        _emit({"id": current_id, "ok": False,
               "error": {"type": type(exc).__name__, "message": str(exc)}})
        print(f"frozen actor bridge failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    return serve(args.checkpoint, args.repo)


if __name__ == "__main__":
    raise SystemExit(main())
