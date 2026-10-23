"""Run one bounded subprocess; retain console output, command and 1 Hz total-GPU samples."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=600)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command or args.timeout < 1:
    parser.error("a command and positive timeout are required")
args.output.mkdir(parents=True, exist_ok=False)
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
begin = time.monotonic()
samples = []
own_samples = []
timed_out = False
with (args.output / "console.txt").open("w") as console, (args.output / "gpu.csv").open("w") as gpu, (args.output / "process_gpu.csv").open("w") as process_gpu:
    writer = csv.writer(gpu)
    writer.writerow(["elapsed_seconds", "gpu_index", "total_used_MiB", "total_MiB", "utilization_percent"])
    process_writer = csv.writer(process_gpu)
    process_writer.writerow(["elapsed_seconds", "pid", "used_MiB", "in_test_process_group"])
    process = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while process.poll() is None:
            try:
                sample = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                for row in csv.reader(sample.stdout.splitlines()):
                    writer.writerow([round(time.monotonic()-begin, 3), *[x.strip() for x in row]])
                    if len(row) == 4:
                        samples.append(int(row[1]))
                gpu.flush()
                process_sample = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                own_memory = 0
                for row in csv.reader(process_sample.stdout.splitlines()):
                    if len(row) != 2:
                        continue
                    pid, used = map(int, row)
                    try:
                        own = os.getpgid(pid) == process.pid
                    except ProcessLookupError:
                        own = False
                    process_writer.writerow([round(time.monotonic()-begin, 3), pid, used, own])
                    if own:
                        own_memory += used
                own_samples.append(own_memory)
                process_gpu.flush()
            except (subprocess.SubprocessError, ValueError):
                pass
            if time.monotonic() - begin > args.timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(1)
        code = process.wait()
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        raise
result = {"started_utc": started, "command": command, "cwd": os.getcwd(), "returncode": code,
          "timeout": timed_out, "duration_seconds": round(time.monotonic()-begin, 3),
          "peak_total_gpu_used_MiB": max(samples, default=None), "peak_test_process_gpu_used_MiB": max(own_samples, default=None),
          "sample_interval_seconds": 1, "measurement_note": "Sampled maxima, total GPU includes desktop; actual interval includes nvidia-smi latency"}
(args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result))
raise SystemExit(124 if timed_out else code)
