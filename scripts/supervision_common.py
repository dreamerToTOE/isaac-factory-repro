"""[ENGINEERING] Linux process identity and atomic supervision evidence."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time

BOOT_ID = Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def stamp():
    return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'unix_seconds': time.time(), 'monotonic_seconds': time.monotonic()}

def identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': pid, 'birth_ticks': fields[19], 'boot_id': BOOT_ID,
                'pgid': int(fields[2]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None

def same_process(expected, live=True):
    if not expected or expected.get('boot_id') != BOOT_ID:
        return False
    actual = identity(expected['pid'])
    return bool(actual and actual['birth_ticks'] == expected.get('birth_ticks')
                and (not live or actual['state'] not in ('Z', 'X')))

def atomic_json(path, value, durable=True, replace=True):
    """Atomic visibility; final evidence is also fsynced. Heartbeats avoid fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush()
            if durable: os.fsync(f.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # atomic O_EXCL equivalent, complete file only
        if durable:
            dfd = os.open(path.parent, os.O_DIRECTORY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass

def read_json(path):
    try:
        result = json.loads(Path(path).read_text())
        return result if isinstance(result, dict) else None
    except (FileNotFoundError, ValueError, OSError):
        return None

def fresh(record, limit):
    if not record or record.get('boot_id') != BOOT_ID:
        return False
    age = time.monotonic() - record.get('monotonic_seconds', -1e30)
    return 0 <= age <= limit

def stop_owned(expected, grace=3):
    """Check birth and group immediately before each signal, including escalation."""
    decisions = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not same_process(expected): return decisions
        actual = identity(expected['pid'])
        if actual['pgid'] != expected['pid']:
            raise RuntimeError('Refusing to signal a process group not owned by child leader')
        try: os.killpg(expected['pid'], sig)
        except ProcessLookupError: return decisions
        decisions.append({'signal': signal.Signals(sig).name, **stamp()})
        until = time.monotonic() + grace
        while time.monotonic() < until:
            if not same_process(expected): return decisions
            time.sleep(.02)
    return decisions

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
