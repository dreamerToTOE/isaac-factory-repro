#!/usr/bin/env python3
"""Read-only numerical reconstruction of archived R2 trajectories; no MuJoCo import.

Contact positions are MuJoCo-reported midpoints: projected peg coordinates do
not establish exact surface point, edge/corner membership, or solver impulse.
"""
import gzip, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'evidence/g4_s/s2a_r2/matrix_v2'
OLD = ROOT/'evidence/g4_s/s2a/run_20261026T140338/seed_1001/closure.jsonl.gz'
OUT = ROOT/'evidence/g4_s/s2a_r2_r1/forensics_v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def records(p):
    with (gzip.open(p,'rt') if p.suffix == '.gz' else p.open()) as f:
        return [json.loads(line) for line in f]
def write(p, value): p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def peg_coordinates(s, c):
    # q = w,x,y,z. Rotation inverse maps world contact midpoint to peg root.
    q=np.array(s['peg_quaternion_wxyz'],dtype=float); q=q/np.linalg.norm(q)
    w,x,y,z=q
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    v=R.T@(np.array(c['position_world_m'])-np.array(s['peg_root_position']))
    return {'xyz_from_peg_root_m':v.tolist(),'axial_distance_to_nearest_end_m':float(min(abs(v[2]),abs(.05-v[2]))),'radial_m':float(np.linalg.norm(v[:2]))}
def contact_summary(s):
    return [dict(c, peg_local_midpoint=peg_coordinates(s,c)) for c in s['contacts']]
def peak(cs):
    return max([(max(0.,-c['distance_m']),c) for c in cs],key=lambda x:x[0],default=(0,None))
def norm_force(cs, side):
    return sum(c['normal_force_n'] for c in cs if 'peg_collision' in (c['geom1'],c['geom2']) and any(f'{side}_finger' in g for g in (c['geom1'],c['geom2'])))
def matrix_summary(m):
    M=np.array(m['physical_full_M']); P=np.array(m['physical_M']); C=np.array(m['controller_M']); inv=np.linalg.inv(M)
    return {'full_shape':list(M.shape),'full_min_eigenvalue':float(np.linalg.eigvalsh(M).min()),'physical_arm_min_eigenvalue':float(np.linalg.eigvalsh(P).min()),'controller_min_eigenvalue':float(np.linalg.eigvalsh(C).min()),'finger_inverse_mass_block':inv[7:9,7:9].tolist(),'physical_controller_frobenius_difference':float(np.linalg.norm(P-C)),'full_symmetry_max_error':float(np.max(abs(M-M.T)))}
def trajrow(r):
    s=r['state']; pre=r['pre_state']; cs=s['contacts']; raw=[-7500*pre['qpos'][i]-173*pre['qvel'][i] for i in (7,8)]; reported=s['finger_command']['raw_pd_force_n']; act=s['integration_actuation']
    return {'step':r['step'],'episode_time_s':r['episode_time_s'],'finger_qpos_m':s['qpos'][7:9],'finger_qvel_m_s':s['qvel'][7:9],
      'finger_pre_qpos_m':pre['qpos'][7:9],'finger_pre_qvel_m_s':pre['qvel'][7:9],
      'reconstructed_raw_pd_n':raw,'reported_raw_pd_n':reported,'raw_pd_reconstruction_error_n':max(abs(a-b) for a,b in zip(raw,reported)),
      'commanded_force_n':s['finger_command']['commanded_force_n'],'integration_actuator_force_n':act['finger_actuator_force_n'],
      'post_forward_finger_force_n':s['finger_actuator_force_n'],'post_forward_generalized_finger_force_n':s['s2a']['finger_generalized_actuator_contribution_n'],
      'contact_normal_force_n':{side:norm_force(cs,side) for side in ('left','right')},'archived_pad_force_n':r['pad_force_n'],
      'contacts':contact_summary(s),'max_penetration_m':peak(cs)[0],
      'peg_hand_slip_translation_m':s['s2a']['peg_hand_slip_m'],'peg_hand_slip_rotation_rad':s['s2a']['peg_hand_slip_rad'],
      'qacc_post_forward':r['qacc_post_forward'],'qfrc_constraint_post_forward':r['qfrc_constraint_post_forward'],
      'qacc_warmstart':r['qacc_warmstart'],'qvel_interval_acceleration':r['qvel_interval_acceleration'],
      'matrix_summary_preintegration':matrix_summary(r['matrices']), 'stop_reasons':r['stop_reasons']}
def preprow(x):
    s=x['state']; cs=s['contacts']; return {'step':x['closure_step'],'finger_qpos_m':s['qpos'][7:9],'finger_qvel_m_s':s['qvel'][7:9],
      'contacts':contact_summary(s),'max_penetration_m':peak(cs)[0],'pad_force_n':x['pad_force_n'],
      'raw_pd_n':s['finger_command'].get('raw_pd_force_n'), 'finger_integration_actuation':s.get('integration_actuation'),
      'relative_peg_hand_m':x['relative_peg_hand_m'],'relative_peg_hand_rotation':x['relative_peg_hand_rotation']}

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    result={'method':'Archived numerical reconstruction only; no new dynamics, MuJoCo import, inference, or viewer.',
      'timing_caveat':'Contact force, qacc and qfrc_constraint are poststep mj_forward recomputations, not previous integration impulses. Integration actuation carries the previous preintegration timestamp.',
      'edge_caveat':'Positions are reported contact midpoints transformed to peg frame. Edge/corner membership and finger-surface feature identity cannot be proven from these values; exact historical collision solver states and manifolds unavailable.',
      'equality_solver_caveat':'qfrc_constraint is aggregate constraint force after forward, not separately attributed equality/contact forces; per-constraint solver impulses/Jacobians/iterations are not archived.',
      'inputs':{},'preparation':{},'trajectories':{}}
    for i in (1,2,3):
        p=OLD if i==1 else BASE/f'initials/I{i}/closure.jsonl'; result['inputs'][str(p.relative_to(ROOT))]=sha(p)
        rows=[preprow(x) for x in records(p)]; write(OUT/f'I{i}_preparation.json',rows)
        maxrow=max(rows,key=lambda r:r['max_penetration_m']); thresholds=[r['step'] for r in rows if r['max_penetration_m']>50e-6]
        result['preparation'][f'I{i}']={'rows':len(rows),'max_penetration_m':maxrow['max_penetration_m'],'max_step':maxrow['step'],'max_contacts':[c for c in maxrow['contacts'] if abs(-c['distance_m']-maxrow['max_penetration_m'])<1e-12],
          'steps_above_50um':thresholds,'final_max_penetration_m':rows[-1]['max_penetration_m'],'final_pad_force_n':rows[-1]['pad_force_n'],
          'reset_admission':'PASS (historical; not retroactively changed)','contact_quality_warning':bool(thresholds)}
    for i in (1,2,3):
      for k in ('B','D'):
        rid=f'I{i}_{k}'; p=BASE/f'trajectories/{rid}/physics.jsonl.gz'; result['inputs'][str(p.relative_to(ROOT))]=sha(p)
        raw=[x for x in records(p) if x['event']=='physics_result']; rows=[trajrow(x) for x in raw]
        with gzip.open(OUT/f'{rid}_timeline.jsonl.gz','wt') as f:
            for x in rows: f.write(json.dumps(x,allow_nan=False,separators=(',',':'))+'\n')
        firstloss=next((x['step'] for x in rows if min(x['contact_normal_force_n'].values())<=0),None)
        result['trajectories'][rid]={'steps':len(rows),'first_any_pad_force_nonpositive':firstloss,'max_penetration_m':max(x['max_penetration_m'] for x in rows),
          'max_raw_pd_reconstruction_error_n':max(x['raw_pd_reconstruction_error_n'] for x in rows),'final':rows[-1],
          'first_state':rows[0], 'loss_window':[x for x in rows if firstloss is not None and firstloss-2<=x['step']<=firstloss+1]}
    write(OUT/'forensics.json',result)
    print(json.dumps({'preparation':result['preparation'],'trajectories':{k:{s:v[s] for s in ('steps','first_any_pad_force_nonpositive','max_penetration_m','max_raw_pd_reconstruction_error_n')} for k,v in result['trajectories'].items()}},indent=2))
if __name__=='__main__': main()
