"""Reproduce compact I2 diagnostics and conventional scientific plots from raw runs.

All simulation execution is elsewhere. This program is CPU-only; full episode
state/statistics remain in the external streaming-export artifact, referenced by
hash rather than replaced by a compact table.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.g4i2_analysis import summarize_actions,summarize_controller,summarize_episodes


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def collect(manifest, output, episode_export=None):
    m=json.loads(Path(manifest).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    runs=[Path(v['output']) for k,v in m['phases'].items() if k.startswith('b_')]
    actions=[];controllers=[];episodes=[];times={};results=[]
    for run in runs:
        result=json.loads((run/'result.json').read_text());assert result['status']=='COMPLETED'
        results.append(result);episodes.extend(json.loads((run/'episodes.json').read_text()))
        times[run.name]=[]
        with (run/'i2_trace.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['kind']=='action':actions.append(row)
                elif row['kind']=='controller':controllers.append(row)
                elif row['kind']=='timing':times[run.name].append(row)
    summary={'actions':summarize_actions(actions),'controller':summarize_controller(controllers),
             'evaluation':summarize_episodes(episodes),'runs':[{k:r[k] for k in ['seed','num_envs','policy_steps','denominator','ever_success','terminal_success','episode_loop_wall_s','auxiliary_physics_s_during_loop']} for r in results]}
    # The detailed full-state episode document is external, including every
    # scientific observation and its statistics, not silently reduced here.
    if episode_export:
        p=Path(episode_export);summary['full_episode_export']={'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    compact=[]
    for ep in episodes:
        state=ep['pre_reset_terminal_state'];delta=np.asarray(state['held_base_pos'])-state['target_held_base_pos']
        compact.append({k:ep[k] for k in ['evaluation_seed','env_id','episode_id','steps','ever_success','terminal_success','first_success_time_seconds','termination_reason','failure_class','trace']}|
                       {'terminal_xy_error_m':float(np.linalg.norm(delta[:2])),'terminal_z_displacement_m':float(delta[2]),'full_state_and_statistics':summary.get('full_episode_export','UNKNOWN')})
    summary['evaluation'].pop('episodes')
    write(out/'summary.json',summary);write(out/'episodes.json',compact)
    with (out/'episodes.csv').open('x',newline='') as stream:
        fields=list(compact[0]);writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for ep in compact:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in ep.items()})
    return summary,actions,times


def plots(summary,actions,times,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(exist_ok=False)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(output/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    selected={name:[r for r in rows if not r['warmup']] for name,rows in times.items()}
    gpu=[];e2e=[];intervals={}
    for name,rows in selected.items():
        gpu.extend(s['device_ms'] for r in rows for s in r['event_sections'] if s['stage']=='forward')
        e2e.extend(r['timings_ms']['observation_to_command'] for r in rows)
        stamps=[r['wall_command_monotonic_ns'] for r in rows];intervals[name]=np.diff(stamps)/1e6
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,values,label in zip(axes,[gpu,e2e],['GPU same-stream forward','Observation to first effort target submission']):
        x=np.sort(values);ax.plot(x,np.arange(1,len(x)+1)/len(x));ax.set(xlabel='ms',ylabel='Empirical CDF',title=label)
        for percentile in [50,95,99]:ax.axvline(np.percentile(x,percentile),alpha=.3,ls='--')
    save(fig,'gpu_and_end_to_end_latency')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,values in intervals.items():axes[0].hist(values,bins=60,histtype='step',label=name)
    axes[0].axvline(1000/15,color='red',ls='--',label='66.67ms simulated policy interval');axes[0].set(xlabel='Actual wall interval (ms)',ylabel='Count',yscale='log');axes[0].legend(fontsize=7)
    names=list(intervals);hz=[1000/np.mean(intervals[n]) for n in names]
    axes[1].bar(names,hz);axes[1].axhline(15,color='red',ls='--',label='15 simulated Hz');axes[1].set(ylabel='Actual wall command Hz',ylim=(0,16));axes[1].legend()
    save(fig,'wall_cadence_and_deadlines')
    raw=np.concatenate([np.asarray(r['raw_mu']) for r in actions]);fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].boxplot([raw[:,i] for i in range(6)],labels=['x','y','z','rx','ry','rz'],showfliers=False);axes[0].axhline(1,color='r',ls='--');axes[0].axhline(-1,color='r',ls='--');axes[0].set(ylabel='Raw mu',title='Original action distribution')
    axes[1].bar(['x','y','z','rx','ry','rz'],[r['clipped_at_unit_rate']*100 for r in summary['actions']['dimensions']]);axes[1].set(ylabel='Saturated samples (%)',title='abs(clipped action) >= 1; all science samples')
    save(fig,'action_saturation')
    fig,ax=plt.subplots(figsize=(9,4));joint=summary['controller']['joints'];x=np.arange(7)
    ax.bar(x-.18,[j['software_changed_rate']*100 for j in joint],.36,label='Software clip changed requested torque')
    ax.bar(x+.18,[j['requested_above_configured_limit_rate']*100 for j in joint],.36,label='Requested above configured effort limit')
    ax.set(xticks=x,xticklabels=[str(i+1) for i in x],xlabel='Arm joint',ylabel='Samples (%)',title='Requested effort is not measured applied effort');ax.legend(fontsize=7)
    save(fig,'torque_diagnostics')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episode-export',type=Path);p.add_argument('--plots',action='store_true');a=p.parse_args()
    summary,actions,times=collect(a.manifest,a.output,a.episode_export)
    if a.plots:plots(summary,actions,times,a.output/'plots')

if __name__=='__main__':main()
