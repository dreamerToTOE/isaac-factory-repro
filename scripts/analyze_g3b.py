"""CPU-only G3B trajectory analysis; original and diagnostic pools never merge."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .compare_g3b import compare_evaluations
    from .diagnostic_metrics import episode_summary, quaternion_axis_angle_deg, scalar_stats
except ImportError:  # Direct script invocation.
    from compare_g3b import compare_evaluations
    from diagnostic_metrics import episode_summary, quaternion_axis_angle_deg, scalar_stats


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def extract_episodes(arrays, expected_horizon=None):
    """Extract complete contiguous episodes, dropping only an unfinished tail.

    An ID returning after another episode or a policy counter reset within one
    ID is rejected rather than concatenated into a manufactured trajectory.
    """
    if 'policy_step' not in arrays or np.asarray(arrays['policy_step']).ndim != 2:
        raise ValueError('policy_step requires vectorized [T,N] shape')
    steps,envs=arrays['policy_step'].shape
    if not steps or not envs:
        raise ValueError('trajectory contains no vectorized samples')
    for name,values in arrays.items():
        if np.asarray(values).ndim < 2 or values.shape[:2] != (steps,envs):
            raise ValueError('%s does not share the [T,N] trajectory dimensions'%name)
    for env in range(envs):
        ids=np.asarray(arrays['episode_id'][:,env])
        boundaries=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,steps]
        seen=set()
        for start,end in zip(boundaries[:-1],boundaries[1:]):
            ep=int(ids[start])
            if ep in seen:
                raise ValueError('episode ID returned after a reset')
            seen.add(ep)
            tr={k:v[start:end,env] for k,v in arrays.items()}
            if not np.array_equal(tr['policy_step'],np.arange(1,end-start+1)):
                raise ValueError('episode policy-step reset or missing samples')
            if 'env_id' in tr and not np.all(tr['env_id']==env):
                raise ValueError('vectorized environment index mismatch')
            done=np.asarray(tr['done'])
            if not np.isin(done,[False,True]).all() or np.any(done[:-1]):
                raise ValueError('done must occur only at an episode final sample')
            if not done[-1]:
                if end != steps:
                    raise ValueError('incomplete episode before another reset')
                continue
            if expected_horizon is not None and len(done)!=expected_horizon:
                raise ValueError('completed episode horizon mismatch')
            yield tr


def last_window_motion_summary(trajectory, window_samples=20):
    """Tail commands and measured motion; no contact mechanism is inferred."""
    if window_samples < 1:
        raise ValueError('window_samples must be positive')
    n=len(trajectory['policy_step']);start=max(0,n-window_samples)
    result={'window_samples':n-start,'first_policy_step':int(trajectory['policy_step'][start]),
            'last_policy_step':int(trajectory['policy_step'][-1]),
            'official_ema_action_rms':None,'policy_pre_ema_action_rms':None,
            'ee_displacement_m':None,'ee_sampled_path_length_m':None,
            'ee_linear_speed_m_s':None,'ee_angular_speed_rad_s':None,
            'interpretation':'Commands and measured motion describe behavior; neither stagnation nor active commands establish contact obstruction.'}
    for source,target in [('actions','official_ema_action_rms'),('policy_actions','policy_pre_ema_action_rms')]:
        if source in trajectory:
            values=np.asarray(trajectory[source][start:],dtype=float)
            valid=values[np.isfinite(values).all(axis=1)]
            result[target]=float(np.sqrt(np.mean(valid**2))) if valid.size else None
    for source,target in [('ee_lin_vel_w','ee_linear_speed_m_s'),('ee_ang_vel_w','ee_angular_speed_rad_s')]:
        if source in trajectory:
            values=np.asarray(trajectory[source][start:],dtype=float)
            norms=np.linalg.norm(values,axis=1);norms[~np.isfinite(values).all(axis=1)]=np.nan
            result[target]=scalar_stats(norms)
    if 'ee_pos_w' in trajectory:
        positions=np.asarray(trajectory['ee_pos_w'][start:],dtype=float)
        if np.isfinite(positions[[0,-1]]).all():
            result['ee_displacement_m']=float(np.linalg.norm(positions[-1]-positions[0]))
        if np.isfinite(positions).all():
            result['ee_sampled_path_length_m']=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
    return result


def select_success_controls(successful, seeds=(1001,1002,1003)):
    return [next(s for s in successful if s['seed']==seed) for seed in seeds if any(s['seed']==seed for s in successful)]


def summarize_secondary(successful):
    def vals(section,key):
        return [s[section][key]['final'] for s in successful if s[section][key]['final'] is not None]
    world_z=vals('geometry','axial_displacement_m')
    axis_gap=vals('secondary','hole_axis_projected_base_gap_m')
    tilt=vals('orientation','axis_angle_deg')
    duration=[s['timing']['sampled_success_duration_s'] for s in successful if s['timing']['sampled_success_duration_s'] is not None]
    return {'scope':'SECONDARY, conditional on newly observed official terminal success only; not a replacement success rate',
            'official_terminal_success_episodes':len(successful),
            'terminal_world_z_below0_count':sum(z<0 for z in world_z),
            'terminal_hole_axis_gap_below_minus1mm_count':sum(z<-.001 for z in axis_gap),
            'terminal_axis_angle_above5deg_count':sum(x>5 for x in tilt),
            'measurement_denominators':{'terminal_world_z':len(world_z),'terminal_hole_axis_gap':len(axis_gap),
                                        'terminal_axis_angle':len(tilt),'sampled_success_duration':len(duration)},
            'sampled_success_duration_s':{'median':float(np.median(duration)) if duration else None,
                                         'min':min(duration) if duration else None,'max':max(duration) if duration else None},
            'interpretation':'Negative base-plane gap is a geometric observation, not independently measured collision penetration. No contact obstruction conclusion.'}


def validate_completion_and_provenance(meta,result,embedded,seed,checkpoint_sha256):
    """Reject partial/failed captures before a completed denominator is reported."""
    for name,value in [('metadata',meta),('metrics',result),('embedded metadata',embedded)]:
        if value.get('status')!='complete':
            raise ValueError('%s is not complete'%name)
        if int(value.get('seed',-1))!=seed or value.get('checkpoint_sha256')!=checkpoint_sha256:
            raise ValueError('%s provenance mismatch'%name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=False)
    original={};new={};summaries=[];trajectories={};provenance=[]
    for run in m['runs']:
        seed=int(run['seed']);meta=json.loads(Path(run['metadata']).read_text());result=json.loads(Path(run['metrics']).read_text())
        original[seed]=json.loads(Path(run['original_g2_metrics']).read_text());new[seed]=result
        if sha(run['trajectory'])!=meta['trajectory']['sha256']:raise ValueError('trajectory digest mismatch')
        with np.load(run['trajectory'],allow_pickle=False) as data:
            embedded=json.loads(str(data['metadata_json']))
            validate_completion_and_provenance(meta,result,embedded,seed,m['checkpoint_sha256'])
            arrays={k:data[k] for k in data.files if k!='metadata_json'}
        if arrays['policy_step'].shape!=(447,32):raise ValueError('Expected447x32 pre-reset records')
        actual={(x['env_id'],x['episode_id']):x for x in result['episodes']}
        if len(actual)!=len(result['episodes']):raise ValueError('duplicate evaluator episode records')
        extracted_keys=set()
        for tr in extract_episodes(arrays,expected_horizon=149):
            s=episode_summary(tr,policy_dt_s=meta['step_dt_seconds'],
                secondary_axis_threshold_deg=m['secondary_metrics']['axis_angle_threshold_deg'],
                plausible_z_range_m=tuple(m['secondary_metrics']['hole_axis_gap_range_m']))
            s['last20_motion']=last_window_motion_summary(tr)
            key=(seed,s['env_id'],s['episode_id']);episode_key=(s['env_id'],s['episode_id'])
            if episode_key not in actual:raise ValueError('trajectory episode absent from evaluator records')
            extracted_keys.add(episode_key);record=actual[episode_key]
            if s['samples']!=149 or not np.all(tr['timeout'][-1]):raise ValueError('Episode horizon/reset mismatch')
            if (s['timing']['ever_success'],s['timing']['terminal_success'],s['timing']['first_success_step']) != (record['ever_success'],record['terminal_success'],record['first_success_step']):raise ValueError('Trajectory/evaluator outcome mismatch')
            if s['geometry']['official_predicate_disagreement_samples']:raise ValueError('Official predicate disagreement')
            # Actual official engagement return, not a contact measurement.
            engage=(tr['xy_error_m']<.0025)&(tr['axial_displacement_m']<meta['fixed_asset_height_m']*meta['official_engage_threshold'])
            if not np.array_equal(engage,tr['engaged']):raise ValueError('Engagement predicate mismatch')
            summaries.append(s);trajectories[key]=tr
        if extracted_keys!=actual.keys():raise ValueError('evaluator completed records absent from trajectories')
        if len(extracted_keys)!=result['episodes_completed']:raise ValueError('completed trajectory denominator mismatch')
        provenance.append({'seed':seed,'trajectory':run['trajectory'],'trajectory_sha256':sha(run['trajectory']),
                           'metadata_sha256':sha(run['metadata']),'metrics_sha256':sha(run['metrics']),
                           'shape':list(arrays['policy_step'].shape)})
    comparison=compare_evaluations(original,new)
    failed=[s for s in summaries if s['timing']['terminal_success'] is False]
    counts={c:sum(any(x['category']==c for x in s['failure_categories']) for s in failed) for c in 'ABCDEF'}
    successful=[s for s in summaries if s['timing']['terminal_success'] is True]
    secondary=summarize_secondary(successful)
    summary={'comparison':comparison,'failure_count':len(failed),'failure_category_counts_overlapping':counts,
             'episodes':summaries,'secondary':secondary,'provenance':provenance,
             'validation':{'trajectory_episode_outcomes_match_evaluator':True,'official_predicate_disagreements':0,
                           'engagement_disagreements':0,'all_episode_lengths149':True,
                           'pre_reset_records':sum(s['samples'] for s in summaries)},
             'analysis_script_sha256':sha(__file__),'metrics_script_sha256':sha(Path(__file__).with_name('diagnostic_metrics.py'))}
    write(a.output/'summary.json',summary);write(a.output/'comparison.json',comparison);write(a.output/'failure_cases.json',failed)
    with (a.output/'episode_summary.csv').open('w',newline='') as f:
        cols=['seed','env_id','episode_id','ever_success','terminal_success','first_success_step','success_duration_s','xy_min_mm','xy_final_mm','z_min_mm','z_final_mm','tilt_min_deg','tilt_final_deg','tilt_max_deg','action_rms','ee_path_mm','last20_ema_action_rms','last20_ee_displacement_mm','last20_ee_mean_speed_m_s','categories']
        w=csv.writer(f);w.writerow(cols)
        for s in summaries:
            w.writerow([s['seed'],s['env_id'],s['episode_id'],s['timing']['ever_success'],s['timing']['terminal_success'],s['timing']['first_success_step'],s['timing']['sampled_success_duration_s'],
              s['geometry']['xy_error_m']['min']*1000,s['geometry']['xy_error_m']['final']*1000,s['geometry']['axial_displacement_m']['min']*1000,s['geometry']['axial_displacement_m']['final']*1000,
              s['orientation']['axis_angle_deg']['min'],s['orientation']['axis_angle_deg']['final'],s['orientation']['axis_angle_deg']['max'],s['actions']['official_ema_actions']['rms_all_components'],
              s['motion']['ee_sampled_path_length_m']*1000,s['last20_motion']['official_ema_action_rms'],
              s['last20_motion']['ee_displacement_m']*1000 if s['last20_motion']['ee_displacement_m'] is not None else None,
              s['last20_motion']['ee_linear_speed_m_s']['mean'] if s['last20_motion']['ee_linear_speed_m_s'] is not None else None,','.join(x['category'] for x in s['failure_categories'])])
    # Small representative traces for reviewing plots without the full external NPZ.
    selected=[(s['seed'],s['env_id'],s['episode_id']) for s in failed]
    for cand in select_success_controls(successful):
        selected.append((cand['seed'],cand['env_id'],cand['episode_id']))
    traces=a.output/'representative_traces';traces.mkdir()
    for key in selected:
        tr=trajectories[key];tilt=quaternion_axis_angle_deg(tr['peg_quat_w'],tr['hole_quat_w'])
        with (traces/('seed%d_env%d_episode%d.csv'%key)).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['policy_step','xy_mm','z_mm','axis_tilt_deg','success','engaged','ee_speed_m_s',*['policy_a%d'%j for j in range(6)],*['ema_a%d'%j for j in range(6)]])
            for i in range(149):w.writerow([int(tr['policy_step'][i]),tr['xy_error_m'][i]*1000,tr['axial_displacement_m'][i]*1000,tilt[i],int(tr['official_success'][i]),int(tr['engaged'][i]),np.linalg.norm(tr['ee_lin_vel_w'][i]),*tr['policy_actions'][i],*tr['actions'][i]])
    plots(a.output,summaries,trajectories,counts)
    print(json.dumps({'comparison':comparison,'failure_count':len(failed),'categories':counts,'secondary':secondary},indent=2))


def plots(out,summaries,trajectories,counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'pdf.fonttype':42,'savefig.bbox':'tight'})
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=170)
        plt.close(fig)
    failures=[s for s in summaries if s['timing']['terminal_success'] is False]
    success=[s for s in summaries if s['timing']['terminal_success'] is True]
    controls=select_success_controls(success)
    if not summaries:raise ValueError('no complete episodes available for plots')
    # Every new failure; successful controls are not matched counterfactuals.
    for field,label,threshold,scale,filename in [('xy_error_m','Official world-XY error (mm)',2.5,1000,'xy_trajectories'),('axial_displacement_m','Official one-sided world-Z gap (mm)',1.,1000,'axial_trajectories'),('tilt','Peg/hole +Z axis angle (deg); secondary',5.,1,'orientation_trajectories')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for s in controls:
            tr=trajectories[(s['seed'],s['env_id'],s['episode_id'])]
            y=quaternion_axis_angle_deg(tr['peg_quat_w'],tr['hole_quat_w']) if field=='tilt' else tr[field]*scale
            ax.plot(tr['policy_step'],y,color='gray',alpha=.3,lw=1)
        for s in failures:
            key=(s['seed'],s['env_id'],s['episode_id']);tr=trajectories[key]
            y=quaternion_axis_angle_deg(tr['peg_quat_w'],tr['hole_quat_w']) if field=='tilt' else tr[field]*scale
            ax.plot(tr['policy_step'],y,lw=1.3,label='%d / env%d / ep%d'%key)
        ax.axhline(threshold,color='black',ls='--',lw=1,label='Secondary 5deg' if field=='tilt' else 'Official strict threshold')
        ax.set(xlabel='Policy step (pre-reset)',ylabel=label,title='New diagnostic failures; gray = %d successful controls (at most one per seed)'%len(controls))
        ax.legend(fontsize=7,loc='upper right',ncol=2);save(fig,filename)
    # One representative failure and one successful episode, same seed, distinct starts.
    s=failures[0] if failures else summaries[0]
    control=next((v for v in success if v['seed']==s['seed']),success[0] if success else None)
    fk=(s['seed'],s['env_id'],s['episode_id']);ck=(control['seed'],control['env_id'],control['episode_id']) if control is not None else None
    pairs=[(fk,'tab:red','failure' if failures else 'example')]
    if ck is not None and ck!=fk:pairs.append((ck,'tab:blue','success control'))
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True)
    for j,ax in enumerate(axes.flat):
        for key,color,label in pairs:
            tr=trajectories[key]
            ax.plot(tr['policy_step'],tr['policy_actions'][:,j],color=color,alpha=.45,lw=.8)
            ax.plot(tr['policy_step'],tr['actions'][:,j],color=color,label=label+' EMA')
        ax.set(title='Action channel%d'%j,xlabel='Policy step',ylabel='Controller command')
        ax.legend(fontsize=7)
    fig.suptitle('Policy input (faint) and official EMA; example %s; success control %s'%(fk,ck if ck is not None else 'unavailable'),fontsize=10);fig.tight_layout();save(fig,'policy_actions')
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True)
    fields=[('xy','XY error (mm)'),('z','World-Z gap (mm)'),('tilt','Axis tilt (deg), secondary'),('action','EMA action norm'),('speed','Measured EE linear speed (m/s)'),('success','Official predicate fraction')]
    for ax,(field,label) in zip(axes.flat,fields):
        for group,color,name in [(success,'tab:blue','success'),(failures,'tab:red','terminal failure')]:
            rows=[]
            for s in group:
                tr=trajectories[(s['seed'],s['env_id'],s['episode_id'])]
                rows.append({'xy':lambda:tr['xy_error_m']*1000,'z':lambda:tr['axial_displacement_m']*1000,'tilt':lambda:quaternion_axis_angle_deg(tr['peg_quat_w'],tr['hole_quat_w']),'action':lambda:np.linalg.norm(tr['actions'],axis=-1),'speed':lambda:np.linalg.norm(tr['ee_lin_vel_w'],axis=-1),'success':lambda:tr['official_success'].astype(float)}[field]())
            if rows:
                rows=np.asarray(rows);med=np.median(rows,axis=0);lo,hi=np.percentile(rows,[25,75],axis=0)
                ax.plot(np.arange(1,150),med,color=color,label='%s n=%d'%(name,len(group)));ax.fill_between(np.arange(1,150),lo,hi,color=color,alpha=.2)
        ax.set(xlabel='Policy step',ylabel=label);ax.legend(fontsize=7)
    fig.suptitle('Descriptive median/IQR by observed outcome; shared policy, not causal groups',fontsize=11);fig.tight_layout();save(fig,'success_failure_comparison')
    fig,ax=plt.subplots(figsize=(7.5,4))
    ax.bar(list(counts),list(counts.values()),color=['tab:blue','tab:orange','tab:purple','gray','tab:red','gray'])
    for i,(c,n) in enumerate(counts.items()):ax.text(i,n+.08,str(n),ha='center')
    ax.set(ylabel='New terminal-failure episodes (overlapping)',xlabel='A XY | B axial | C secondary tilt | D contact | E lost success | F unknown',title='Observed conditions; contact mechanism not established',ylim=(0,max(1,max(counts.values()))+1));save(fig,'failure_categories')

if __name__=='__main__':main()
