"""Sacrificial CPU processes: wait status, gated guard, lifecycle and atomicity."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from scripts.supervision_common import atomic_json, identity, read_json, same_process, stop_owned

ROOT=Path(__file__).parents[1]
SUP=ROOT/'scripts/durable_supervisor.py'
GUARD=ROOT/'scripts/gpu_guard.py'
PY=sys.executable

def await_file(path, timeout=20):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        v=read_json(path)
        if v:return v
        time.sleep(.02)
    raise AssertionError(f'No valid JSON at {path}')

def plain(tmp,code):
    out=tmp/'sup'
    p=subprocess.run([PY,str(SUP),'--output',str(out),'--lock-file',str(tmp/'lock'),'--',PY,'-c',code],capture_output=True,text=True,timeout=20)
    return p,await_file(out/'status.json')

def guarded(tmp,code,rows=None):
    mock=tmp/'mock.json';mock.write_text(json.dumps(rows or [{'free_MiB':4096}]))
    sup=tmp/'sup';guard=tmp/'guard'
    g=subprocess.Popen([PY,str(GUARD),'--output',str(guard),'--supervisor-directory',str(sup),'--mock-samples',str(mock)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    s=subprocess.Popen([PY,str(SUP),'--output',str(sup),'--guard-directory',str(guard),'--lock-file',str(tmp/'lock'),
                        '--startup-timeout','3','--',PY,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return s,g,sup,guard

def finish(s,g,sup,guard):
    so,se=s.communicate(timeout=20);go,ge=g.communicate(timeout=20)
    assert not se and not ge,(se,ge)
    return await_file(sup/'status.json'),await_file(guard/'status.json')

def test_normal_child_exit_zero(tmp_path):
    p,s=plain(tmp_path,"print('normal')")
    assert p.returncode==0 and s['raw_wait_returncode']==0 and s['child_exit_code']==0 and s['termination_reason']=='child_exit'
    assert s['finalized'] and s['child_identity']['birth_ticks']

def test_nonzero_child_status_is_not_monitor_status(tmp_path):
    p,s=plain(tmp_path,'raise SystemExit(7)')
    assert p.returncode==7 and s['child_exit_code']==7 and s['termination_reason']=='child_exit'

def test_signal_wait_status_preserved(tmp_path):
    p,s=plain(tmp_path,'import os,signal;os.kill(os.getpid(),signal.SIGTERM)')
    assert p.returncode==124 and s['raw_wait_returncode']==-15 and s['child_exit_code'] is None and s['child_signal']=='SIGTERM'

def test_gated_guard_samples_before_exec_and_through_wait(tmp_path):
    s,g,sp,gp=guarded(tmp_path,'import time;time.sleep(.3)')
    sr,gr=finish(s,g,sp,gp)
    assert s.returncode==g.returncode==0
    assert gr['continuous_monitoring'] and not gr['monitoring_gaps']
    release=read_json(sp/'exec_release.json')['monotonic_seconds']
    child_spawn=read_json(sp/'child_identity.json')['monotonic_seconds']
    waited=read_json(sp/'child_exit.json')['monotonic_seconds']
    assert gr['first_sample']['monotonic_seconds']<child_spawn<release<=waited<=gr['last_sample']['monotonic_seconds']
    assert read_json(gp/'attached_identity.json')['target_identity']==sr['child_identity']

def test_missing_guard_prevents_training_exec(tmp_path):
    out=tmp_path/'sup';marker=tmp_path/'must_not_exist'
    p=subprocess.run([PY,str(SUP),'--output',str(out),'--guard-directory',str(tmp_path/'missing'),
                      '--startup-timeout','.2','--lock-file',str(tmp_path/'lock'),'--',PY,'-c',f"open({str(marker)!r},'w').close()"],capture_output=True,timeout=20)
    s=await_file(out/'status.json')
    assert p.returncode!=0 and s['child_identity'] is None and s['raw_wait_returncode'] is None and not marker.exists()

def test_guard_crash_is_detected_and_child_is_stopped(tmp_path):
    s,g,sp,gp=guarded(tmp_path,'import time;time.sleep(20)')
    await_file(sp/'exec_release.json');g.kill();g.wait()
    s.communicate(timeout=20)
    sr=await_file(sp/'status.json')
    assert sr['termination_reason']=='guard_missing_or_stale' and sr['raw_wait_returncode']<0
    assert not (gp/'status.json').exists()  # abrupt loss is not forged completion

def test_supervisor_crash_is_detected_independently(tmp_path):
    s,g,sp,gp=guarded(tmp_path,'import time;time.sleep(20)')
    await_file(sp/'exec_release.json');s.kill();s.wait()
    g.communicate(timeout=20);gr=await_file(gp/'status.json')
    assert gr['stop_reason']=='supervisor_lost_or_stale' and not gr['continuous_monitoring']
    assert not same_process(gr['target_identity']) and not (sp/'status.json').exists()

def test_three_low_samples_trigger_without_real_gpu_pressure(tmp_path):
    rows=[{'free_MiB':4096}]*8+[{'free_MiB':900}]*3
    s,g,sp,gp=guarded(tmp_path,'import time;time.sleep(20)',rows)
    sr,gr=finish(s,g,sp,gp)
    assert gr['stop_reason']=='gpu_headroom_floor' and gr['max_low_consecutive']==3
    assert sr['raw_wait_returncode']<0 and not gr['continuous_monitoring']

def test_nonconsecutive_low_samples_reset_counter(tmp_path):
    rows=[{'free_MiB':4096}]*8+[{'free_MiB':900},{'free_MiB':900},{'free_MiB':4096}]*2
    s,g,sp,gp=guarded(tmp_path,'import time;time.sleep(1.4)',rows)
    sr,gr=finish(s,g,sp,gp)
    assert sr['child_exit_code']==0 and gr['continuous_monitoring'] and gr['max_low_consecutive']==2

def test_monitor_exception_stops_child_and_records_failure(tmp_path):
    s,g,sp,gp=guarded(tmp_path,'import time;time.sleep(20)',[{'free_MiB':4096}]*8+[{'error':'mock NVML failure'}])
    sr,gr=finish(s,g,sp,gp)
    assert gr['stop_reason']=='monitor_failed' and gr['error'] and not gr['continuous_monitoring']
    assert sr['raw_wait_returncode']<0

def test_sampling_gap_is_explicit_and_not_continuous(tmp_path):
    s,g,sp,gp=guarded(tmp_path,'import time;time.sleep(20)',[{'free_MiB':4096}]*8+[{'free_MiB':4096,'delay_seconds':.65}])
    sr,gr=finish(s,g,sp,gp)
    assert gr['monitoring_gaps'] and gr['stop_reason']=='monitoring_gap' and not gr['continuous_monitoring']

def test_pid_birth_mismatch_does_not_signal_unrelated_process(tmp_path):
    c=subprocess.Popen([PY,'-c','import time;time.sleep(.3)'],start_new_session=True)
    who=identity(c.pid);who['birth_ticks']=str(int(who['birth_ticks'])+1)
    assert stop_owned(who)==[]
    assert c.wait(timeout=5)==0

def test_duplicate_lock_is_held_by_child_even_if_supervisor_dies(tmp_path):
    out=tmp_path/'first'
    executed=tmp_path/'executed.json'
    command=f"import time,json;open({str(executed)!r},'w').write(json.dumps({{'exec_started':True}}));time.sleep(5)"
    s=subprocess.Popen([PY,str(SUP),'--output',str(out),'--lock-file',str(tmp_path/'lock'),'--',PY,'-c',command])
    await_file(executed)  # exec_release record precedes pipe write; wait for actual exec
    await_file(out/'exec_release.json');who=await_file(out/'child_identity.json')['identity']
    s.kill();s.wait()
    p=subprocess.run([PY,str(SUP),'--output',str(tmp_path/'second'),'--lock-file',str(tmp_path/'lock'),'--',PY,'-c','pass'],capture_output=True,timeout=10)
    try:
        assert p.returncode==73 and not (tmp_path/'second').exists()
    finally:stop_owned(who)

def test_existing_evidence_never_overwritten(tmp_path):
    p,s=plain(tmp_path,'pass');before=(tmp_path/'sup/command.json').read_bytes()
    p,s2=plain(tmp_path,'raise SystemExit(6)')
    assert p.returncode==73 and (tmp_path/'sup/command.json').read_bytes()==before

def test_atomic_status_visibility_and_failed_replace_keeps_old(tmp_path,monkeypatch):
    path=tmp_path/'status.json';atomic_json(path,{'n':0,'padding':'x'*1000})
    errors=[];done=threading.Event()
    def reader():
        while not done.is_set():
            try:json.loads(path.read_text())
            except Exception as e:errors.append(repr(e))
    t=threading.Thread(target=reader);t.start()
    for n in range(25):atomic_json(path,{'n':n,'padding':'x'*1000},durable=False)
    done.set();t.join();assert not errors
    old=path.read_bytes()
    def fail(*a):raise OSError('simulated rename failure')
    monkeypatch.setattr(os,'replace',fail)
    try:atomic_json(path,{'n':999})
    except OSError:pass
    assert path.read_bytes()==old

def test_systemd_units_survive_submitting_launcher_sigkill(tmp_path):
    # Only newly-created test units, with a finite sacrificial child.
    mock=tmp_path/'mock.json';mock.write_text('[{"free_MiB":4096}]')
    out=tmp_path/'service';prefix='g3a-cpu-'+uuid.uuid4().hex[:10]
    p=subprocess.Popen([PY,str(ROOT/'scripts/g3a_launch_smoke.py'),'--output',str(out),'--unit-prefix',prefix,
                         '--cwd',str(ROOT),'--mock-samples',str(mock),'--lock-file',str(tmp_path/'lock'),'--wait','--',
                         PY,'-c','import time;time.sleep(.8)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    await_file(out/'supervisor/exec_release.json');p.kill();p.communicate(timeout=5)
    sr=await_file(out/'supervisor/status.json');gr=await_file(out/'guard/status.json')
    assert sr['raw_wait_returncode']==0 and gr['continuous_monitoring']
