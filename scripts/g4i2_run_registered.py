"""Execute only preregistered I2 phases through the unchanged guarded launcher.

Every invocation creates fresh resource evidence and records the real supervisor
wait status. No retries, process killing, training or unguarded fallback exist.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--phase',required=True)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text());phase=manifest['phases'][a.phase]
    raw=Path(manifest['raw_root']);raw.mkdir(parents=True,exist_ok=True)
    if a.phase!='a_events':
        accepted=json.loads((raw/'i2_a_events_acceptance.json').read_text())
        if not accepted['passed']: raise RuntimeError('Mandatory I2-A failed; STOP')
    if a.phase.startswith('b_'):
        if not json.loads((raw/'i2_a_acceptance.json').read_text())['passed']:raise RuntimeError('I2-A acceptance required')
    for name,expected in manifest['runtime_source_sha256'].items():
        if sha(REPO/name)!=expected:raise RuntimeError('Source freeze mismatch: '+name)
    output=Path(phase['output']);supervision=Path(phase['supervision'])
    if output.exists() or supervision.exists():raise FileExistsError('Fresh output paths required; no retry/overwrite')
    resource={}
    for key,cmd in {'gpu':['nvidia-smi','--query-gpu=timestamp,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],
                    'compute':['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],
                    'disk':['df','-B1',str(raw),'/dev/shm']}.items():
        run=subprocess.run(cmd,capture_output=True,text=True);resource[key]={'argv':cmd,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
        if run.returncode:raise RuntimeError('Resource query failed')
    resource.update(utc=dt.datetime.now(dt.timezone.utc).isoformat(),phase=a.phase)
    write(raw/(a.phase+'_resource_preflight.json'),resource)
    free=float(resource['gpu']['stdout'].splitlines()[0].split(',')[6])
    if free<phase['minimum_preflight_free_mib']:raise RuntimeError('Insufficient fresh GPU headroom; STOP, no process killed')
    base=Path(manifest['workspace'])
    cmd=['python3',str(REPO/'scripts/g3a_launch_smoke.py'),'--output',str(supervision),'--unit-prefix',phase['unit_prefix'],
         '--cwd',str(base/'work/runs'),'--timeout',str(phase['timeout_s']),'--wait',
         '--setenv','FACTORY_VENV='+str(base/'work/envs/factory-g01'),'--setenv','ISAACSIM_PATH=/home/ubuntu2004/isaacsim-4.5.0',
         '--setenv','OMP_NUM_THREADS=4','--setenv','FACTORY_G2_AUDIT_OUTPUT=','--setenv','FACTORY_G3A_FINALIZE_TENSORBOARD=',
         '--','bash',str(REPO/'scripts/run_sim.sh'),str(REPO/'scripts/g4i1_runtime.py'),
         '--headless','--device','cuda:0','--i2','--output',str(output),'--seed',str(phase['seed']),
         '--num-envs',str(phase['num_envs']),'--episodes-per-env',str(phase['episodes_per_env']),
         '--max-steps',str(phase['max_steps']),'--warmup','20','--profiling',phase['profiling'],'--diagnostics',phase['diagnostics']]
    if phase['validate']:cmd.append('--validate')
    write(raw/(a.phase+'_command.json'),{'argv':cmd,'manifest_sha256':sha(a.manifest),'utc':resource['utc']})
    start=time.monotonic()
    with (raw/(a.phase+'_launcher_stdout.txt')).open('x') as stdout,(raw/(a.phase+'_launcher_stderr.txt')).open('x') as stderr:
        completed=subprocess.run(cmd,stdout=stdout,stderr=stderr)
    durable=raw/(a.phase+'_supervision');shutil.copytree(supervision,durable)
    supervisor=json.loads((durable/'supervisor/status.json').read_text())
    guard=json.loads((durable/'guard/status.json').read_text())
    result={'phase':a.phase,'launcher_returncode':completed.returncode,'wall_s':time.monotonic()-start,
            'supervisor':supervisor,'guard':guard,'durable_supervision':str(durable),
            'runtime_error_exists':(output/'runtime_error.json').exists(),'result_exists':(output/'result.json').exists()}
    write(raw/(a.phase+'_execution.json'),result)
    print(json.dumps(result,indent=2),flush=True)
    good=(completed.returncode==0 and supervisor['raw_wait_returncode']==0 and supervisor['termination_reason']=='child_exit' and guard['continuous_monitoring'] and not result['runtime_error_exists'] and result['result_exists'])
    return 0 if good else 1

if __name__=='__main__':raise SystemExit(main())
