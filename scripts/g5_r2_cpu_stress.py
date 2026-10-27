"""Bounded CPU-only R2 lifecycle and heartbeat filesystem experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

try:
    from scripts.supervision_common import atomic_json
    from scripts.gpu_guard import HeartbeatPublisher
except ModuleNotFoundError:  # direct execution as scripts/g5_r2_cpu_stress.py
    from supervision_common import atomic_json
    from gpu_guard import HeartbeatPublisher


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return values[index]


def file_benchmark(base, iterations=100):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='g5-r2-heartbeat-', dir=base) as temp:
        path = Path(temp) / 'heartbeat.json'
        durations = []
        for n in range(iterations):
            start = time.monotonic()
            atomic_json(path, {'sample': n, 'payload': 'x' * 2048}, durable=False)
            durations.append(time.monotonic() - start)
        return {'filesystem': str(base), 'iterations': iterations,
                'median_seconds': statistics.median(durations),
                'p95_seconds': percentile(durations, .95),
                'max_seconds': max(durations), 'min_seconds': min(durations)}


def publisher_benchmark(base, iterations=100):
    base = Path(base)
    with tempfile.TemporaryDirectory(prefix='g5-r2-publisher-', dir=base) as temp:
        publisher = HeartbeatPublisher(Path(temp) / 'heartbeat.json', stale_seconds=10)
        submit = []
        for n in range(iterations):
            start = time.monotonic()
            publisher.submit({'sample': n, 'payload': 'x' * 2048}, time.monotonic())
            submit.append(time.monotonic() - start)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = publisher.close(timeout=0.01)
            if status['published_count'] >= 1 and not status['thread_alive_after_close']:
                break
            # close requests stop only after the current bounded write; reopen
            # is intentionally avoided, so the loop exits on the first close.
            break
        else:
            status = publisher.close()
        return {'filesystem': str(base), 'iterations': iterations,
                'submit_median_seconds': statistics.median(submit),
                'submit_p95_seconds': percentile(submit, .95),
                'submit_max_seconds': max(submit),
                'published_count': status['published_count'],
                'coalesced_count': status['coalesced_count'],
                'publisher_failure': status['failure']}


def lifecycle(repo, repetitions):
    py = os.environ.get('PYTHON', 'python3')
    root = Path(repo)
    guard = root / 'scripts' / 'gpu_guard.py'
    supervisor = root / 'scripts' / 'durable_supervisor.py'
    results = []
    with tempfile.TemporaryDirectory(prefix='g5-r2-lifecycle-') as temp:
        temp = Path(temp)
        for index in range(repetitions):
            mock = temp / f'mock-{index}.json'
            mock.write_text(json.dumps([{'free_MiB': 4096}]))
            sup = temp / f'sup-{index}'
            guard_dir = temp / f'guard-{index}'
            started = time.monotonic()
            g = subprocess.Popen([py, str(guard), '--output', str(guard_dir),
                                  '--supervisor-directory', str(sup), '--mock-samples', str(mock)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            s = subprocess.Popen([py, str(supervisor), '--output', str(sup), '--guard-directory',
                                  str(guard_dir), '--startup-timeout', '3', '--lock-file',
                                  str(temp / f'lock-{index}'), '--', py, '-c',
                                  'import time; time.sleep(.15)'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            sout, serr = s.communicate(timeout=20)
            gout, gerr = g.communicate(timeout=20)
            elapsed = time.monotonic() - started
            status = json.loads((sup / 'status.json').read_text())
            monitor = json.loads((guard_dir / 'status.json').read_text())
            results.append({'iteration': index, 'wall_seconds': elapsed,
                            'supervisor_returncode': status['raw_wait_returncode'],
                            'guard_continuous_monitoring': monitor['continuous_monitoring'],
                            'guard_stop_reason': monitor['stop_reason'],
                            'sample_count': monitor['sample_count'],
                            'stderr_empty': not serr and not gerr,
                            'stdout_empty': not sout and not gout})
    return {'repetitions': repetitions, 'results': results,
            'pass_count': sum(x['supervisor_returncode'] == 0 and x['guard_continuous_monitoring'] for x in results),
            'wall_median_seconds': statistics.median(x['wall_seconds'] for x in results),
            'wall_p95_seconds': percentile([x['wall_seconds'] for x in results], .95),
            'wall_max_seconds': max(x['wall_seconds'] for x in results)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repetitions', type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 10 or args.repetitions < 3:
        parser.error('bounded experiment requires at least 10 iterations and 3 repetitions')
    tmpfs = Path(f'/run/user/{os.getuid()}')
    result = {'schema_version': 1,
              'cpu_only': True,
              'mounts': {'repo': str(args.repo), 'tmpfs': str(tmpfs)},
              'atomic_json': [file_benchmark(args.repo / 'work', args.iterations),
                              file_benchmark(tmpfs, args.iterations)],
              'publisher_submit': [publisher_benchmark(args.repo / 'work', args.iterations),
                                   publisher_benchmark(tmpfs, args.iterations)],
              'lifecycle': lifecycle(args.repo, args.repetitions)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
