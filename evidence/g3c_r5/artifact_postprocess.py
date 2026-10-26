"""Durable CPU-only artifact collection, Chinese reporting, and regression after R5.
This script never launches or controls GPU work, training, evaluation, or Git.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def atomic(p,data):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);m=json.loads(a.manifest.read_text());repo=Path(m['repository']);e=repo/'evidence/g3c_r5';s=Path(m['training_supervisor_directory'])/'status.json';c=Path(m['completion_output'])/'status.json';progress={'started_utc':now(),'phase':'waiting_for_training_and_evaluation','cpu_only':True,'training_or_evaluation_control_actions':[],'commands':[],'manifest_sha256':sha(a.manifest),'postprocess_sha256':sha(__file__)}
 def pulse():atomic(a.output/'progress.json',progress)
 def run(command,label,environment=None):
  receipt={'argv':command,'cwd':str(repo),'started_utc':now()};progress.update(phase=label);pulse()
  with (a.output/(label+'.stdout.txt')).open('x') as stdout,(a.output/(label+'.stderr.txt')).open('x') as stderr:
   r=subprocess.run(command,cwd=repo,env=environment,stdout=stdout,stderr=stderr)
  receipt.update(returncode=r.returncode,ended_utc=now());atomic(a.output/(label+'.command.json'),receipt);progress['commands'].append(receipt);pulse()
  if r.returncode:raise RuntimeError(f'CPU step {label} returned {r.returncode}; see {a.output}')
 try:
  if progress['manifest_sha256']!='f3163299d669e355c401cb825661bb99705a3b10b79b1f16fed4b090bfb77103':raise ValueError('Manifest identity changed')
  for p,d in m['frozen_source_sha256'].items():
   if sha(p)!=d:raise ValueError('Frozen source changed: '+p)
  while not (s.is_file() and c.is_file()):pulse();time.sleep(2)
  training=json.loads(s.read_text());completion=json.loads(c.read_text())
  if training.get('finalized') is not True or completion.get('controller_exit_code') is None:raise ValueError('Actual terminal records missing')
  progress['terminal_training_status']=training;progress['completion_phase']=completion.get('phase');pulse()
  run(['/usr/bin/python3',str(repo/'scripts/collect_g3cr5.py'),'--manifest',str(a.manifest),'--output',str(e/'raw_collect')],'collect_evidence')
  run(['/usr/bin/python3',str(repo/'scripts/analyze_g3cr5.py'),'--manifest',str(a.manifest),'--output',str(e/'analysis'),'--tensorboard-python',m['python']],'analyze')
  run(['/usr/bin/python3',str(repo/'scripts/report_g3cr5.py'),'--report',str(e/'analysis/report.json'),'--baseline',m['seed0_analysis_report'],'--manifest',str(a.manifest),'--output',str(repo/'docs/REVIEW_G3C_R5.md'),'--result',str(e/'result.json')],'chinese_report')
  environment=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');environment.pop('PYTHONPATH',None)
  run([m['python'],'-m','pytest','-q','tests','--basetemp=/dev/shm/g3cr5-cpu-final-20261026t015040z','--junitxml='+str(e/'final_cpu_regression.xml')],'cpu_regression',environment)
  progress.update(phase='artifacts_and_regression_complete',ended_utc=now(),exit_code=0);pulse();atomic(a.output/'status.json',progress);return 0
 except BaseException as error:
  progress.update(phase='incomplete',ended_utc=now(),exit_code=1,error={'type':type(error).__name__,'message':str(error)});pulse();atomic(a.output/'status.json',progress);traceback.print_exc();return 1
if __name__=='__main__':raise SystemExit(main())
