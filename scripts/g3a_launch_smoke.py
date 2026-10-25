#!/usr/bin/env python3
"""[ENGINEERING] Submit prestarted guard and gated supervisor as separate services.

No unit restarts training. The submitting client may disappear after submission;
systemd owns both units. Use fresh paths. Training argv is passed unchanged.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from supervision_common import atomic_json, read_json, sha, stamp


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--unit-prefix',required=True)
    p.add_argument('--cwd',type=Path,required=True)
    p.add_argument('--lock-file',type=Path,default=Path(f'/run/user/{os.getuid()}/isaac-factory-gpu0.lock'))
    p.add_argument('--setenv',action='append',default=[])
    p.add_argument('--timeout',type=float,default=1200)
    p.add_argument('--wait',action='store_true')
    p.add_argument('--mock-samples',type=Path,help='CPU sacrificial lifecycle tests only')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(argv);command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or not a.unit_prefix or not all(c.isalnum() or c=='-' for c in a.unit_prefix):p.error('Command and alphanumeric/hyphen prefix required')
    if a.timeout<=0 or not a.cwd.is_dir():p.error('Positive timeout and existing cwd required')
    from durable_supervisor import SAFE_ENV
    if any(x.split('=',1)[0] not in SAFE_ENV or '=' not in x for x in a.setenv):p.error('Only documented safe environment keys supported')
    a.output=a.output.resolve();a.cwd=a.cwd.resolve();a.lock_file=a.lock_file.resolve();scripts=Path(__file__).resolve().parent
    a.output.mkdir(parents=True,exist_ok=False)
    units={role:a.unit_prefix+'-'+role for role in ('guard','supervisor')}
    sup=a.output/'supervisor';guard=a.output/'guard'
    guard_cmd=[sys.executable,str(scripts/'gpu_guard.py'),'--output',str(guard),'--supervisor-directory',str(sup)]
    if a.mock_samples:guard_cmd+=['--mock-samples',str(a.mock_samples.resolve())]
    sup_cmd=[sys.executable,str(scripts/'durable_supervisor.py'),'--output',str(sup),'--guard-directory',str(guard),
             '--lock-file',str(a.lock_file),'--timeout',str(a.timeout),'--',*command]
    def unit(role,cmd):
        return ['systemd-run','--user','--collect','--unit='+units[role],
                '--property=WorkingDirectory='+str(a.cwd),'--property=Restart=no','--property=KillMode=control-group',
                '--property=TimeoutStopSec=10',*(['--setenv='+v for v in a.setenv] if role=='supervisor' else []),*cmd]
    plan={'schema_version':2, 'created':stamp(),'training_command':command,'cwd':str(a.cwd),
          'safe_environment':a.setenv,'units':units,'unit_commands':{'guard':unit('guard',guard_cmd),'supervisor':unit('supervisor',sup_cmd)},
          'source_sha256':{str(scripts/f):sha(scripts/f) for f in ('supervision_common.py','supervised_exec.py','durable_supervisor.py','gpu_guard.py','g3a_launch_smoke.py')}}
    atomic_json(a.output/'launch.json',plan,replace=False)
    for role in ('guard','supervisor'):
        result=subprocess.run(plan['unit_commands'][role],capture_output=True,text=True)
        atomic_json(a.output/(role+'_submission.json'),{'role':role,'returncode':result.returncode,
                    'stdout':result.stdout,'stderr':result.stderr,**stamp()},replace=False)
        if result.returncode:return result.returncode
    print(str(a.output),flush=True)
    if not a.wait:return 0
    until=time.monotonic()+a.timeout+60
    while time.monotonic()<until:
        s=read_json(sup/'status.json');g=read_json(guard/'status.json')
        if s and g:
            return 0 if s['raw_wait_returncode']==0 and s['termination_reason']=='child_exit' and g['continuous_monitoring'] else 124
        time.sleep(.2)
    return 124
if __name__=='__main__':raise SystemExit(main())
