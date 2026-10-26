"""Offline-only, lossless S2A evidence bundle; never loads actor or simulator."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def collect(raw,output,process,console,resources):
    raw,output=Path(raw).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((raw/'artifact_manifest.json').read_text())
    for p,digest in manifest.items():
        if sha(p)!=digest:raise ValueError('RAW_DIGEST_MISMATCH:'+p)
    summary=json.loads((raw/'summary.json').read_text())
    protocol=json.loads((raw/'preregistered_protocol.json').read_text())
    source_checks={name:sha(ROOT/name)==digest for name,digest in protocol['source_files_sha256'].items()}
    if not all(source_checks.values()):raise ValueError('AT_RUN_SOURCE_CHANGED')
    seeds=[x['seed'] for x in summary['episodes']]
    if seeds != [1001,1002,1003][:len(seeds)]:raise ValueError('UNREGISTERED_OR_REPEATED_SEEDS')
    counts=[]
    for result in summary['episodes']:
        directory=raw/f"seed_{result['seed']}"
        read=lambda name:[json.loads(x) for x in (directory/name).read_text().splitlines()]
        closure=read('closure.jsonl') if (directory/'closure.jsonl').exists() else []
        physics=read('physics.jsonl') if (directory/'physics.jsonl').exists() else []
        policy=read('policy.jsonl') if (directory/'policy.jsonl').exists() else []
        inference=read('inference.jsonl') if (directory/'inference.jsonl').exists() else []
        requests=sum(x['event']=='request' for x in inference)
        responses=sum(x['event']=='response' for x in inference)
        if (len(physics)!=result.get('physics_steps_recorded',result['physics_steps_completed'])
                or len(policy)!=result['policy_steps_completed'] or responses!=result['policy_inferences']):
            raise ValueError('TRACE_COUNT_MISMATCH')
        counts.append({'seed':result['seed'],'closure_lines':len(closure),'physics_lines':len(physics),
                       'policy_lines':len(policy),'inference_requests':requests,'inference_responses':responses,
                       'full_length_complete':result['policy_steps_completed']==149,'technical_pass':result.get('technical_pass',False)})
    output.mkdir(parents=True,exist_ok=False)
    sources=sorted(p for p in raw.rglob('*') if p.is_file())
    sources.extend([Path(process),Path(console),Path(resources)])
    index=[]
    for p in sources:
        relative=p.relative_to(raw) if p.is_relative_to(raw) else Path(p.name)
        target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        compressed=p.suffix=='.jsonl'
        if compressed:target=Path(str(target)+'.gz')
        if target.exists():raise FileExistsError(target)
        if compressed:
            with p.open('rb') as src,target.open('xb') as dst,gzip.GzipFile(fileobj=dst,mode='wb',mtime=0,filename='') as gz:
                shutil.copyfileobj(src,gz)
            with gzip.open(target,'rb') as f:
                if hashlib.sha256(f.read()).hexdigest()!=sha(p):raise ValueError('LOSSLESS_COMPRESSION_FAILED')
        else:shutil.copyfile(p,target)
        index.append({'raw_path':str(p),'raw_sha256':sha(p),'raw_bytes':p.stat().st_size,
                      'bundle_path':str(target.relative_to(ROOT)),'bundle_sha256':sha(target),
                      'bundle_bytes':target.stat().st_size,'lossless_gzip':compressed})
    receipt={'task':'G4-S2A','operation':'OFFLINE_LOSSLESS_EVIDENCE_COLLECTION','raw_directory':str(raw),
             'raw_manifest_verified':True,'runtime_sources_unchanged':source_checks,
             'status':summary['status'],'counts':counts,'artifacts':index,
             'total_raw_bytes':sum(x['raw_bytes'] for x in index),'total_bundled_bytes':sum(x['bundle_bytes'] for x in index),
             'policy_or_physics_executions':0,'collector_sha256':sha(Path(__file__))}
    write(output/'collection.json',receipt)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','output','process','console','resources'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();r=collect(a.raw,a.output,a.process,a.console,a.resources)
    print(json.dumps({k:v for k,v in r.items() if k not in ('artifacts','runtime_sources_unchanged')}))

if __name__=='__main__':main()
