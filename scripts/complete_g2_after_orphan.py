"""[ENGINEERING] Observe the existing G2 process, then evaluate saved milestones.

This never starts/resumes training. A reattached independent memory guard must
already be active. The original child exit code remains unknown; completion is
assessed from the uninterrupted epoch audit, final save, and official marker.
Launch via systemd-run --user so chat/terminal lifecycle cannot kill the parent.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--evaluation-amendment',type=Path,required=True)
    p.add_argument('--training-pid',type=int,required=True)
    p.add_argument('--training-birth-ticks',required=True)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text());amendment=json.loads(args.evaluation_amendment.read_text())
    ws=Path(manifest['workspace']);repo=Path(manifest['repository']);root=Path(manifest['monitor_root'])
    status_path=root/'continuation_status.json'
    if status_path.exists():raise RuntimeError('Refusing to overwrite continuation evidence')
    status={'phase':'observing_existing_training','started_utc':now(),'pid':os.getpid(),
            'training_pid':args.training_pid,'training_birth_ticks':args.training_birth_ticks,
            'manifest_sha256':sha(args.manifest),'evaluation_amendment_sha256':sha(args.evaluation_amendment),
            'training_exit_code':None,'training_exit_code_note':'Original parent monitor disappeared; exit status is unavailable.',
            'training_restarted':False,'evaluations':[]}
    save(status_path,status)
    proc=Path(f'/proc/{args.training_pid}/stat')
    def alive():
        try:
            fields=proc.read_text().split()
            return fields[21]==args.training_birth_ticks and fields[2]!='Z'
        except FileNotFoundError:return False
    def verify_evaluator():
        for file,digest in amendment['frozen_evaluation_sha256'].items():
            assert sha(repo/file)==digest,f'Evaluator changed after amendment: {file}'
    verify_evaluator()
    env=dict(os.environ,FACTORY_VENV=str(ws/'work/envs/factory-g01'),ISAACSIM_PATH='/home/ubuntu2004/isaacsim-4.5.0',OMP_NUM_THREADS='4')
    for key in ('FACTORY_AUDIT_OUTPUT','FACTORY_G2_AUDIT_OUTPUT','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'):env.pop(key,None)
    py=ws/'work/envs/factory-g01/bin/python'
    try:
        while alive():time.sleep(2)
        status['training_disappearance_observed_utc']=now()
        status['phase']='validating_training_evidence';save(status_path,status)
        for _ in range(50):
            if (root/'reattached_guard/result.json').exists():break
            time.sleep(.1)
        guard=json.loads((root/'reattached_guard/result.json').read_text())
        assert guard['stop_reason'] is None,guard
        audit=json.loads((root/'train/training_audit.json').read_text())
        console=(root/'train/console.txt').read_text()
        assert [e['epoch'] for e in audit['epochs']]==list(range(1,201))
        assert sum(e['environment_transitions'] for e in audit['epochs'])==1638400
        assert all(e['epoch_loss_arrays_finite'] for e in audit['epochs'])
        assert not audit['instrumentation_errors']
        assert not any(e['operation']=='restore' for e in audit['checkpoint_events'])
        assert 'MAX EPOCHS NUM!' in console
        status['training_completed_200_epochs']=True
        status['training_completion_basis']='Same PID/birth throughout observation; contiguous epoch 1..200 audit, 1,638,400 transitions, official MAX EPOCHS NUM marker, checkpoint epoch/frame verification. Exit code remains unknown.'
        status['training_end_observation']=guard['finished_utc']
        run=ws/'work/runs/logs/rl_games/Factory'/manifest['run_name']
        inventory=[{'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted((run/'nn').glob('*.pth'))]
        save(root/'checkpoint_inventory.json',inventory)
        selected={};prior=None
        for epoch in (50,100,150,200):
            candidates=sorted((run/'nn').glob(f'last_Factory_ep_{epoch}_rew_*.pth'))
            assert candidates,f'No epoch {epoch} checkpoint'
            ckpt=next((c for c in candidates if '_rew__' in c.name),candidates[0])
            cmd=[str(py),str(repo/'scripts/inspect_checkpoint.py'),str(ckpt)]
            if prior:cmd+=['--compare',str(prior)]
            checked=subprocess.run(cmd,env=env|{'OMP_NUM_THREADS':'2'},text=True,capture_output=True)
            (root/f'checkpoint_inspection_epoch{epoch}.stderr.txt').write_text(checked.stderr)
            assert checked.returncode==0,checked.stderr
            data=json.loads(checked.stdout)
            assert data['epoch']==epoch and data['frame']==epoch*8192 and data['all_tensors_finite']
            selected[str(epoch)]=data;prior=ckpt
        save(root/'selected_checkpoints.json',selected)
        status['phase']='evaluation';save(status_path,status)
        for epoch in (50,100,150,200):
            ckpt=Path(selected[str(epoch)]['path']);assert sha(ckpt)==selected[str(epoch)]['sha256']
            episodes=3 if epoch==200 else 1
            for seed in manifest['evaluation']['seeds']:
                verify_evaluator()
                output=root/'evaluation'/f'epoch{epoch:03d}'/f'seed{seed}'
                cmd=[sys.executable,str(repo/'scripts/run_bounded.py'),'--output',str(output),'--timeout','1800','--min-free-mib','1024','--',
                     'bash',str(repo/'scripts/run_sim.sh'),str(repo/'scripts/evaluate_checkpoint.py'),'--headless','--device','cuda:0',
                     '--task','Isaac-Factory-PegInsert-Direct-v0','--checkpoint',str(ckpt),'--num_envs','32','--episodes',str(episodes),
                     '--seed',str(seed),'--max_steps',str(160*episodes),'--snapshot-limit','12','--progress-interval','10','--output',str(output/'metrics.json')]
                status.update(current_evaluation={'epoch':epoch,'seed':seed,'episodes_requested':32*episodes},active_command=cmd)
                save(status_path,status)
                result=subprocess.run(cmd,cwd=ws/'work/runs',env=env|{'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'})
                assert result.returncode==0,f'Evaluation failed: epoch={epoch}, seed={seed}, returncode={result.returncode}'
                metrics=json.loads((output/'metrics.json').read_text())
                assert metrics['episodes_completed']==32*episodes
                status['evaluations'].append({'epoch':epoch,'seed':seed,'output':str(output/'metrics.json'),'aggregate':metrics['aggregate']})
                save(status_path,status)
        status['phase']='training_and_evaluation_complete'
    except BaseException as error:
        status['phase']='incomplete';status['error']=repr(error);traceback.print_exc();raise
    finally:
        status['finished_utc']=now();save(status_path,status)


if __name__=='__main__':main()
