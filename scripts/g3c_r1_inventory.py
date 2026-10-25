#!/usr/bin/env python3
"""Read-only GPU/system inventory for G3C-R1.

This script never starts, stops, or signals a process. It samples NVML through
nvidia-smi and records process/system snapshots alongside the samples.
"""
from __future__ import annotations
import argparse,csv,datetime,json,subprocess,time
from pathlib import Path

def command(argv):
    p=subprocess.run(argv,text=True,capture_output=True)
    return {'argv':argv,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

def meminfo():
    result={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value,*rest=line.split()
        if rest and rest[0]=='kB': result[key]=int(value)*1024
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=60);ap.add_argument('--interval',type=float,default=.1)
    a=ap.parse_args()
    if a.duration<60 or a.interval<=0:ap.error('duration must be at least 60 seconds and interval positive')
    a.output.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    snapshots={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duration_requested_s':a.duration,'interval_requested_s':a.interval,'gpu_identity':command(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits']),'driver_cuda_identity':command(['nvidia-smi']),'compute_processes_start':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),'processes_start':command(['ps','-eo','pid,ppid,etimes,stat,cmd']),'systemd_failed_start':command(['systemctl','--user','--no-pager','--plain','--failed','--type=service']),'ram_start':meminfo()}
    rows=[];next_t=start
    with (a.output/'gpu_samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['elapsed_s','timestamp','name','driver_version','memory_total_MiB','memory_used_MiB','memory_free_MiB','memory_reserved_MiB','utilization_gpu_pct','temperature_gpu_C']);writer.writeheader()
        while time.monotonic()-start<a.duration:
            q=command(['nvidia-smi','--query-gpu=timestamp,name,driver_version,memory.total,memory.used,memory.free,memory.reserved,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'])
            now=time.monotonic();line=q['stdout'].strip().splitlines()[0] if q['returncode']==0 and q['stdout'].strip() else ''
            parts=[x.strip() for x in line.split(',')]
            row={'elapsed_s':now-start,'timestamp':parts[0] if len(parts)>0 else None,'name':parts[1] if len(parts)>1 else None,'driver_version':parts[2] if len(parts)>2 else None,'memory_total_MiB':float(parts[3]) if len(parts)>3 else None,'memory_used_MiB':float(parts[4]) if len(parts)>4 else None,'memory_free_MiB':float(parts[5]) if len(parts)>5 else None,'memory_reserved_MiB':float(parts[6]) if len(parts)>6 else None,'utilization_gpu_pct':float(parts[7]) if len(parts)>7 else None,'temperature_gpu_C':float(parts[8]) if len(parts)>8 else None}
            rows.append(row);writer.writerow(row);f.flush();next_t=max(next_t+a.interval,time.monotonic());time.sleep(max(0,next_t-time.monotonic()))
    snapshots.update({'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sample_count':len(rows),'gpu_samples_csv':str((a.output/'gpu_samples.csv').resolve()),'compute_processes_end':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),'processes_end':command(['ps','-eo','pid,ppid,etimes,stat,cmd']),'systemd_failed_end':command(['systemctl','--user','--no-pager','--plain','--failed','--type=service']),'ram_end':meminfo(),'observed':{'peak_used_MiB':max((x['memory_used_MiB'] for x in rows if x['memory_used_MiB'] is not None),default=None),'min_free_MiB':min((x['memory_free_MiB'] for x in rows if x['memory_free_MiB'] is not None),default=None),'max_reserved_MiB':max((x['memory_reserved_MiB'] for x in rows if x['memory_reserved_MiB'] is not None),default=None),'max_utilization_pct':max((x['utilization_gpu_pct'] for x in rows if x['utilization_gpu_pct'] is not None),default=None),'min_utilization_pct':min((x['utilization_gpu_pct'] for x in rows if x['utilization_gpu_pct'] is not None),default=None)}})
    (a.output/'inventory.json').write_text(json.dumps(snapshots,indent=2,sort_keys=True)+'\n')
    print(json.dumps(snapshots['observed'],sort_keys=True))
if __name__=='__main__':main()
