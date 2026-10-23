"""Bound one process group; sample NVML at 10 Hz and stop on low GPU headroom.

The memory guard terminates only the launched process group. It does not modify
the simulator or training configuration. CSV maxima are sampled, not exact peaks.
"""
import argparse
import ctypes as C
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


class Memory(C.Structure):
    _fields_ = [("version", C.c_uint)] + [(name, C.c_ulonglong) for name in ("total", "reserved", "free", "used")]


class ProcessInfo(C.Structure):
    _fields_ = [("pid", C.c_uint), ("usedGpuMemory", C.c_ulonglong),
                ("gpuInstanceId", C.c_uint), ("computeInstanceId", C.c_uint)]


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--min-free-mib", type=int, default=1024)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0 or args.timeout <= 0 or args.min_free_mib < 0:
        parser.error("command, positive interval/timeout and nonnegative memory floor required")
    # Use the driver's read-only NVML interface; no added Python dependency.
    nvml = C.CDLL("libnvidia-ml.so.1")
    assert nvml.nvmlInit_v2() == 0
    device = C.c_void_p()
    assert nvml.nvmlDeviceGetHandleByIndex_v2(C.c_uint(0), C.byref(device)) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    samples = []
    errors = []
    reason = None
    process = None
    try:
        with (args.output / "console.txt").open("w") as console, (args.output / "gpu.csv").open("w") as gpu:
            writer = csv.writer(gpu)
            writer.writerow(["elapsed_seconds", "total_used_MiB", "free_MiB", "total_MiB", "test_compute_used_MiB", "driver_reserved_MiB"])
            # Check available headroom before launching the workload.
            memory = Memory()
            memory.version = C.sizeof(memory) | (2 << 24)
            assert nvml.nvmlDeviceGetMemoryInfo_v2(device, C.byref(memory)) == 0
            if memory.free / 2**20 < args.min_free_mib:
                raise RuntimeError("GPU headroom is below the requested floor before launch")
            process = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
            next_sample = begin
            low_samples = 0
            while process.poll() is None:
                now = time.monotonic()
                status = nvml.nvmlDeviceGetMemoryInfo_v2(device, C.byref(memory))
                if status:
                    errors.append({"elapsed": now-begin, "nvml_memory_status": status})
                    reason = "memory_monitor_failed"
                    stop(process)
                    break
                count = C.c_uint(64)
                infos = (ProcessInfo * 64)()
                status = nvml.nvmlDeviceGetComputeRunningProcesses_v2(device, C.byref(count), infos)
                own_memory = 0
                if status == 0:
                    for item in infos[:count.value]:
                        try:
                            if os.getpgid(item.pid) == process.pid and item.usedGpuMemory < 2**63:
                                own_memory += item.usedGpuMemory
                        except ProcessLookupError:
                            pass
                else:
                    errors.append({"elapsed": now-begin, "nvml_process_status": status})
                    own_memory = None
                row = [now-begin, memory.used/2**20, memory.free/2**20, memory.total/2**20,
                       None if own_memory is None else own_memory/2**20, memory.reserved/2**20]
                samples.append(row)
                writer.writerow([round(x, 3) if x is not None else "" for x in row])
                gpu.flush()
                low_samples = low_samples + 1 if row[2] < args.min_free_mib else 0
                if low_samples >= 3:
                    reason = "gpu_headroom_floor"
                    stop(process)
                    break
                if now-begin > args.timeout:
                    reason = "timeout"
                    stop(process)
                    break
                next_sample = max(next_sample + args.interval, time.monotonic())
                time.sleep(max(0, next_sample-time.monotonic()))
            code = process.wait()
    except BaseException:
        if process is not None:
            stop(process)
        raise
    finally:
        nvml.nvmlShutdown()
    intervals = [b[0]-a[0] for a, b in zip(samples, samples[1:])]
    result = {"started_utc": started, "command": command, "cwd": os.getcwd(),
              "returncode": code, "stop_reason": reason, "duration_seconds": time.monotonic()-begin,
              "sample_count": len(samples), "requested_interval_seconds": args.interval,
              "median_actual_interval_seconds": statistics.median(intervals) if intervals else None,
              "max_actual_interval_seconds": max(intervals, default=None),
              "peak_total_gpu_used_MiB": max((r[1] for r in samples), default=None),
              "min_free_MiB": min((r[2] for r in samples), default=None),
              "peak_test_compute_used_MiB": max((r[4] for r in samples if r[4] is not None), default=None),
              "min_free_mib_guard": args.min_free_mib, "monitor_errors": errors,
              "measurement_note": "NVML v2 sampled at requested 10 Hz; used excludes driver-reserved memory (like nvidia-smi), and includes desktop and other apps. Free is reported separately; used+free+reserved=total. Compute usage belongs only to the launched process group. Sampling can miss shorter peaks; PyTorch allocator peaks are recorded separately."}
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(code if reason is None else 124)


if __name__ == "__main__":
    main()
