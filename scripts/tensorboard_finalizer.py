"""[ENGINEERING] Finalize the RL-Games TensorBoard writer at normal train exit.

RL-Games 1.6.1 creates a ``tensorboardX.SummaryWriter`` inside
``A2CBase.__init__`` and returns from ``agent.train()`` without an explicit
``flush()``/``close()``.  The Isaac Lab training entry then closes the
simulator.  This module wraps only ``Runner.run_train`` so that the writer
owned by the newly-created agent is flushed and closed in a ``finally`` block.
It does not touch the PPO update loop, task, reward, physics, seed, or config.

The wrapper is intentionally opt-in.  The normal repository entry point calls
:func:`install_runner_finalizer` before executing the upstream script; unit
tests can exercise :func:`wrap_runner_class` with a fake Runner and writer.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar


_T = TypeVar("_T")


def finalize_writer(writer: Any) -> dict[str, Any]:
    """Flush then close one writer, returning an auditable operation record.

    ``IntervalSummaryWriter`` in RL-Games delegates ``flush``/``close`` via
    ``__getattr__`` to its underlying tensorboardX writer, so the same code
    covers both the direct and interval writer cases.  A missing writer is
    explicit in the result.  Exceptions are recorded rather than replacing a
    training exception raised by the wrapped ``run_train`` call.
    """

    result: dict[str, Any] = {
        "writer_present": writer is not None,
        "writer_class": type(writer).__name__ if writer is not None else None,
        "flush_called": False,
        "close_called": False,
        "errors": [],
        "success": False,
    }
    if writer is None:
        return result

    for method_name in ("flush", "close"):
        try:
            method = getattr(writer, method_name)
            method()
            result[f"{method_name}_called"] = True
        except Exception as error:  # pragma: no cover - exercised by a test fake
            result["errors"].append({"operation": method_name, "error": repr(error)})
    result["success"] = bool(
        result["writer_present"]
        and result["flush_called"]
        and result["close_called"]
        and not result["errors"]
    )
    return result


def wrap_runner_class(runner_class: type[_T]) -> type[_T]:
    """Install the finalizer around ``Runner.run_train`` once.

    The original method still creates the agent, restores a checkpoint, and
    runs the official algorithm.  We temporarily wrap only the factory's
    ``create`` method to retain a reference to that agent, then restore the
    factory method before finalizing.  This keeps the patch independent of
    RL-Games' private agent subclasses while preserving its original method
    body and return value.
    """

    marker = "_factory_tensorboard_finalizer_installed"
    if getattr(runner_class, marker, False):
        return runner_class

    original_run_train = runner_class.run_train

    @functools.wraps(original_run_train)
    def run_train_with_finalizer(self: Any, args: Any) -> Any:
        agents: list[Any] = []
        factory = self.algo_factory
        original_create = factory.create

        @functools.wraps(original_create)
        def create_and_capture(*positional: Any, **keyword: Any) -> Any:
            agent = original_create(*positional, **keyword)
            agents.append(agent)
            return agent

        factory.create = create_and_capture
        train_error = None
        try:
            return original_run_train(self, args)
        except BaseException as error:
            train_error = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            # Restore the factory even when the official train method raises.
            factory.create = original_create
            records = []
            for agent in agents:
                record = finalize_writer(getattr(agent, "writer", None))
                record["global_rank"] = int(getattr(agent, "global_rank", 0))
                record["writer_expected"] = record["global_rank"] == 0
                records.append(record)
            # A rank-zero agent without a writer, or a call that never creates
            # an agent, cannot establish normal summary finalization.
            finalizer_error = not records or any(
                record["writer_expected"] and not record["success"] for record in records)
            payload = {"status": "error" if train_error or finalizer_error else "complete", "agents": records,
                       "training_error": train_error, "no_agent_created": not records}
            print("FACTORY_TENSORBOARD_FINALIZER=" + json.dumps(payload, sort_keys=True), flush=True)
            output = os.environ.get("FACTORY_G3A_FINALIZER_OUTPUT")
            if output:
                path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
                    os.replace(temporary, path)
                finally:
                    try: os.unlink(temporary)
                    except FileNotFoundError: pass

    runner_class.run_train = run_train_with_finalizer
    setattr(runner_class, marker, True)
    return runner_class


def install_runner_finalizer() -> type[Any]:
    """Patch the installed RL-Games Runner and return its class."""

    from rl_games.torch_runner import Runner

    return wrap_runner_class(Runner)
