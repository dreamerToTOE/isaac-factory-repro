"""CPU-only S0 validation/preservation receipt. No simulation entry point."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[1]
START='280634679633bf127e13349741379b3d38a27a1f'
FACTORY_PYTHON=REPO.parents[1]/'work/envs/factory-g01/bin/python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    with path.open('x') as stream:
        stream.write(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def snapshot():
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',START],cwd=REPO,text=True).splitlines()
    changed=subprocess.check_output(['git','diff','--name-only',START,'--',*files],cwd=REPO,text=True).splitlines()
    if changed:raise RuntimeError('Historical tracked files changed:'+repr(changed))
    audit=json.loads((REPO/'evidence/g4_i/i0/checkpoint_audit.json').read_text())
    external={}
    for label,entry in audit['source_files'].items():
        value=sha(entry['path']);assert value==entry['sha256'],label
        external[entry['path']]=value
    checkpoints={}
    for label,entry in audit['checkpoints'].items():
        value=sha(entry['path']);assert value==entry['sha256'],label
        checkpoints[label]={'path':entry['path'],'sha256':value,'bytes':Path(entry['path']).stat().st_size}
        for stem in ('agent','env'):
            path=entry[stem+'_config_path'];value=sha(path)
            assert value==entry[stem+'_config_sha256'],path
            external[path]=value
    other=REPO.parent/'isaac-factory-repro'
    unrelated={}
    for path in [other/'docs/DEMO_G3V.md',other/'scripts/g3v_playback.sh',*sorted((other/'evidence/g3v').rglob('*'))]:
        if path.is_file():unrelated[str(path)]=sha(path)
    return {'tracked':{name:{'sha256':sha(REPO/name),'executable':bool((REPO/name).stat().st_mode&0o111)} for name in files},
            'frozen_external_sources_and_configurations':external,'checkpoints':checkpoints,
            'unrelated_original_checkout_g3v':unrelated}


def tests(output,python,targets,label):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONDONTWRITEBYTECODE':'1'}
    base=Path(tempfile.mkdtemp(prefix='g4s0_cpu_',dir='/dev/shm' if Path('/dev/shm').is_dir() else None))/'pytest'
    junit=output/(label+'.xml')
    command=[str(python),'-m','pytest','-q',*targets,'-p','no:cacheprovider','--basetemp='+str(base),'--junitxml='+str(junit)]
    start=time.monotonic()
    result=subprocess.run(command,cwd=REPO,env=env,capture_output=True,text=True,timeout=600)
    with (output/(label+'.txt')).open('x') as stream:stream.write(result.stdout+result.stderr)
    cases=ET.parse(junit).findall('.//testcase')
    skipped=sum(case.find('skipped') is not None for case in cases)
    failed=sum(case.find('failure') is not None or case.find('error') is not None for case in cases)
    return {'command':command,'exit_code':result.returncode,'passed':len(cases)-skipped-failed,
            'skipped':skipped,'failed':failed,'duration_s':time.monotonic()-start,
            'log':str((output/(label+'.txt')).relative_to(REPO)),'junit':str(junit.relative_to(REPO))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();before=snapshot()
    write(output/'preservation_before.json',before)
    dependency={name:importlib.util.find_spec(name) is not None for name in ('mujoco','torch','numpy','pxr')}
    full=tests(output,FACTORY_PYTHON,['tests'],'full_cpu')
    plot=tests(output,sys.executable,['tests/test_g4i2_analysis.py::test_plots_are_real_files_and_cannot_overwrite'],'system_plot')
    after=snapshot();write(output/'preservation_after.json',after)
    assert before==after,'Protected contents changed during CPU validation'
    receipts={}
    for name in ('assets_audit.json','controller_audit.json','observation_success_audit.json'):
        path=REPO/'evidence/g4_s/s0'/name
        data=json.loads(path.read_text())
        receipts[name]={'path':str(path.relative_to(REPO)),'sha256':sha(path)}
    passed=full['exit_code']==0 and plot['exit_code']==0 and full['failed']==plot['failed']==0
    summary={'task':'G4-S0','starting_commit':START,'engineering_status':'COMPLETE' if passed else 'CPU_VALIDATION_FAILED',
             'decision':'CONDITIONAL','sim2sim_status':'NOT_RUN_NOT_ASSESSED','full_cpu':full,'system_plot':plot,
             'current_python':sys.executable,'python_version':platform.python_version(),
             'default_python_dependency_availability':dependency,'factory_python':str(FACTORY_PYTHON),
             'preservation':{'equal_before_after':True,'tracked_files':len(before['tracked']),
                             'checkpoint_count':len(before['checkpoints']),
                             'unrelated_g3v_file_count':len(before['unrelated_original_checkout_g3v'])},
             'audit_receipts':receipts,'duration_s':time.monotonic()-start,
             'restrictions':{'physics_steps':0,'rollouts':0,'training':False,'packages_installed':False,
                             'checkpoint_modified':False,'s1_started':False,'s2_started':False},
             'unresolved':['exact Factory asset conversion/public redistribution license scope',
                           'complete table/visual dependency closure',
                           'SDF/concave insertion geometry with bounded conversion error',
                           'runtime mass-overridden inertia and body +0.01I mapping',
                           'raw quaternion sign, TCP/finger Jacobian/M parity',
                           'finger mimic/contact grasp, actual actuator effort/limits',
                           'cross-engine reset/random streams and contact solver differences'],
             'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in [REPO/'scripts/audit_g4s0.py',REPO/'scripts/g4s0_mapping.py',REPO/'scripts/g4s0_asset_audit.py',REPO/'tests/test_g4s0_mapping.py']}}
    write(output/'summary.json',summary)
    print(json.dumps({'engineering_status':summary['engineering_status'],'decision':'CONDITIONAL',
                      'full_cpu':full,'system_plot':plot,'preservation':summary['preservation']},indent=2))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
