#!/usr/bin/env python3
"""[ENGINEERING] Own child wait/exit evidence; release exec only after guard binds.

Run under a user systemd service (Restart=no). A separate guard samples before
spawn. The small gated child keeps its PID/birth across exec and inherits the
exclusive lock, so supervisor loss cannot silently allow duplicate training.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from supervision_common import BOOT_ID, atomic_json, fresh, identity, read_json, same_process, stamp, stop_owned

SAFE_ENV = ('FACTORY_VENV', 'ISAACSIM_PATH', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES',
            'FACTORY_G2_AUDIT_OUTPUT', 'FACTORY_G3A_FINALIZE_TENSORBOARD', 'FACTORY_G3A_FINALIZER_OUTPUT')


def run(args):
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    args.output = args.output.resolve()
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(args.lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock); raise RuntimeError('duplicate launch prevented: device lock is held')
    try: args.output.mkdir(parents=True, exist_ok=False)
    except BaseException:
        os.close(lock); raise
    who = identity(); began = stamp(); child = None; child_id = None; released = None
    reason = None; error = None; code = None; gate_write = None; signals = []; guard_id = None
    seen_signal = []
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    for s in handlers: signal.signal(s, lambda n, frame: seen_signal.append(n))
    def pulse(state):
        atomic_json(args.output/'heartbeat.json', {'boot_id': BOOT_ID, **stamp(), 'identity': who,
                    'child_identity': child_id, 'state': state}, durable=False)
    def event(name, **data):
        with (args.output/'events.jsonl').open('a') as f:
            import json
            f.write(json.dumps({'event': name, **stamp(), **data}, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    def guard_ok(hb, bound=False):
        return bool(hb and fresh(hb, args.stale_seconds) and same_process(hb.get('identity'))
                    and hb.get('sample_count', 0)>0 and hb.get('free_MiB', 0)>=0
                    and hb.get('state') in (('bound',) if bound else ('ready', 'bound'))
                    and (not bound or hb.get('target_identity') == child_id))
    metadata = {'schema_version': 2, 'engineering': True, 'command': command, 'cwd': os.getcwd(),
                'environment': {k:os.environ[k] for k in SAFE_ENV if k in os.environ},
                'identity': who, 'boot_id': BOOT_ID, 'started': began,
                'lock_file': str(args.lock_file), 'guard_directory': str(args.guard_directory) if args.guard_directory else None,
                'stale_seconds': args.stale_seconds, 'startup_timeout_seconds': args.startup_timeout,
                'timeout_seconds': args.timeout}
    atomic_json(args.output/'command.json', metadata, replace=False)
    atomic_json(args.output/'identity.json', {'identity':who, 'boot_id':BOOT_ID, **began}, replace=False)
    try:
        pulse('starting')
        if args.guard_directory:
            until = time.monotonic()+args.startup_timeout
            while True:
                pulse('waiting_for_guard')
                if seen_signal: raise RuntimeError('supervisor signalled before launch')
                hb=read_json(args.guard_directory/'heartbeat.json')
                if guard_ok(hb):
                    if hb.get('free_MiB', 0)<1024: raise RuntimeError('guard preflight headroom below1024MiB')
                    guard_id=hb['identity']; event('guard_ready', guard_identity=guard_id); break
                if read_json(args.guard_directory/'status.json'): raise RuntimeError('guard failed before child spawn')
                if time.monotonic()>until: raise RuntimeError('guard readiness timeout; child not launched')
                time.sleep(.03)
        with (args.output/'console.txt').open('x') as console:
            gate_read, gate_write=os.pipe()
            os.set_inheritable(lock, True)
            child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('supervised_exec.py')),str(gate_read),*command],
                                   stdout=console,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(gate_read,lock))
            os.close(gate_read)
            child_id=identity(child.pid)
            if not child_id: raise RuntimeError('child birth identity unavailable')
            child_spawned=stamp()
            atomic_json(args.output/'child_identity.json', {'identity':child_id,'boot_id':BOOT_ID, **child_spawned},replace=False)
            pulse('gated'); event('child_spawned_gated', identity=child_id)
            if args.guard_directory:
                until=time.monotonic()+args.startup_timeout
                while True:
                    pulse('gated')
                    if seen_signal: raise RuntimeError('supervisor signalled while gated')
                    hb=read_json(args.guard_directory/'heartbeat.json')
                    if guard_ok(hb, True) and hb['identity']==guard_id: break
                    if read_json(args.guard_directory/'status.json'): raise RuntimeError('guard failed while child gated')
                    if time.monotonic()>until: raise RuntimeError('guard target binding timeout')
                    time.sleep(.02)
            released=stamp()
            atomic_json(args.output/'exec_release.json', {'identity':child_id, 'boot_id':BOOT_ID, **released},replace=False)
            os.write(gate_write,b'G'); os.close(gate_write); gate_write=None
            event('exec_released', guard_identity=guard_id)
            while child.poll() is None:
                pulse('running')
                if seen_signal:
                    reason='supervisor_signal'; signals=stop_owned(child_id); break
                if time.monotonic()-began['monotonic_seconds']>args.timeout:
                    reason='timeout'; signals=stop_owned(child_id); break
                if args.guard_directory:
                    hb=read_json(args.guard_directory/'heartbeat.json')
                    guard_status=read_json(args.guard_directory/'status.json')
                    if guard_status:
                        reason=guard_status.get('stop_reason','guard_failed'); signals=stop_owned(child_id); break
                    if not guard_ok(hb, True) or hb['identity']!=guard_id:
                        reason='guard_missing_or_stale'; signals=stop_owned(child_id); break
                time.sleep(args.poll)
            code=child.wait()
            # Publish wait immediately; guard keeps sampling until this record.
            waited=stamp()
            atomic_json(args.output/'child_exit.json', {'identity':child_id, 'raw_wait_returncode':code,
                        'boot_id':BOOT_ID, **waited},durable=False,replace=False)
            if reason is None: reason='child_exit' if code>=0 else 'child_signal'
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)}; reason=reason or 'supervisor_error'
        if child:
            signals=stop_owned(child_id) if child_id else []
            code=child.wait()
            if not (args.output/'child_exit.json').exists():
                atomic_json(args.output/'child_exit.json', {'identity':child_id,'raw_wait_returncode':code,
                            'boot_id':BOOT_ID, **stamp()},durable=False,replace=False)
    finally:
        if gate_write is not None: os.close(gate_write)
        final={'schema_version':2,'finalized':True,'identity':who,'child_identity':child_id,
               'command':command,'started':began,'ended':stamp(),'exec_released':released,
               'wall_seconds':time.monotonic()-began['monotonic_seconds'],
               'raw_wait_returncode':code,'child_exit_code':code if code is not None and code>=0 else None,
               'child_signal':signal.Signals(-code).name if code is not None and code<0 else None,
               'termination_reason':reason,'received_supervisor_signals':seen_signal,'stop_signals':signals,
               'error':error,'guard_identity':guard_id,'guard_required':bool(args.guard_directory)}
        pulse('finished'); atomic_json(args.output/'status.json',final,replace=False)
        for s,h in handlers.items():signal.signal(s,h)
        os.close(lock)
    return 0 if code==0 and reason=='child_exit' and error is None else (code if code and code>0 else 124)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--guard-directory',type=Path)
    p.add_argument('--lock-file',type=Path,default=Path(f'/run/user/{os.getuid()}/isaac-factory-gpu0.lock'))
    p.add_argument('--stale-seconds',type=float,default=2)
    p.add_argument('--startup-timeout',type=float,default=30)
    p.add_argument('--timeout',type=float,default=43200)
    p.add_argument('--poll',type=float,default=.05)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(argv)
    if not a.command or min(a.stale_seconds,a.startup_timeout,a.timeout,a.poll)<=0:p.error('command and positive timing required')
    try:return run(a)
    except (RuntimeError,FileExistsError) as e:print(str(e),file=sys.stderr);return 73

if __name__=='__main__':raise SystemExit(main())
