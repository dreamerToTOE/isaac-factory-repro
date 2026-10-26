#!/usr/bin/env python3
"""At most three fresh, gated frozen-PPO episodes; never imported by regressions.

No historical source/model is modified. A prepared-state view reuses the exact
S1B reset mathematics without resetting the physically closed independent peg.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.g4s1b_contract import ReferenceRunner, axis_angle, quat_conjugate, quat_mul, quat_from_euler
from scripts.g4s1b_env import AdaptedPegInsertEnv, EnvironmentConfig, MENAGERIE_COMMIT, model_xml

START = '5b39da565b4e579aaf600739d2e8fb180222a5da'
CHECKPOINT_SHA = 'f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f'
XML_SHA = 'c60c33ee11eb4806294aee259f97d987308b23e83da62f2219cbc9950fb195c9'
SEEDS = (1001, 1002, 1003)
MAX_POLICY_STEPS, DECIMATION, PHYSICS_DT, MAX_RUNTIME_S = 149, 8, 1/120, 1200
SOURCE_RESET_Q = np.array([.0015178, -.19651, -.0014364, -1.9761, -.00027717, 1.7796, .78556])
CHECKPOINT = Path('/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/runs/logs/rl_games/Factory/g2_peginsert_64env_seed0_baseline_rerun/nn/last_Factory_ep_200_rew__350.4961_.pth')
ACTOR_PYTHON = '/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/envs/factory-g01/bin/python'
ENDPOINT_TOLERANCE_M = 1e-5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value): return {'nonfinite_witness':repr(value)}
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {k: jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(x) for x in value]
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def line(stream, value):
    stream.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)+'\n')
    stream.flush()


def verify_identity(checkpoint_path, xml_text):
    if sha(checkpoint_path) != CHECKPOINT_SHA: raise ValueError('CHECKPOINT_IDENTITY_GATE_FAILED')
    if hashlib.sha256(xml_text.encode()).hexdigest() != XML_SHA: raise ValueError('XML_IDENTITY_GATE_FAILED')
    return True


def verify_artifacts(manifest):
    for path,expected in manifest.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f'ARTIFACT_INTEGRITY_FAILED:{path}')
    return True


def finite_tree(value):
    if isinstance(value, dict): return all(finite_tree(x) for x in value.values())
    if isinstance(value, (tuple, list, np.ndarray)): return all(finite_tree(x) for x in value)
    if isinstance(value, (int, float, np.number)): return bool(np.isfinite(value))
    return True


def validate_policy_result(result):
    for key, shape in {'observation':(19,), 'normalized':(19,), 'mu':(6,), 'clipped':(6,),
                       'h':(2,1,1024), 'c':(2,1,1024)}.items():
        a = np.asarray(result[key])
        if a.shape != shape or not np.isfinite(a).all(): raise ValueError(f'INVALID_POLICY_{key}')
    if not result['rms_unchanged']: raise ValueError('FROZEN_RMS_CHANGED')
    if not np.array_equal(np.asarray(result['clipped']), np.clip(result['mu'], -1., 1.)):
        raise ValueError('ACTION_CLIPPING_MISMATCH')
    return True


def admission_errors(m):
    errors=[]
    for key in ('bilateral_last10','finite','independent_peg','starts_above_hole'):
        if not m[key]: errors.append(key)
    if not np.isfinite(m['stability_m']) or m['stability_m'] > .001: errors.append('translation_stability')
    if not np.isfinite(m['angular_stability_rad']) or m['angular_stability_rad'] > .05: errors.append('rotation_stability')
    if not np.isfinite(m['q_limit_violation']) or m['q_limit_violation'] > 1e-6: errors.append('joint_limits')
    if m['interference_bodies']: errors.append('non_fingertip_interference')
    return errors


def contact_quality(diag, max_assembly_penetration):
    distance=diag.get('minimum_contact_distance_m',diag.get('min_contact_distance_m',0.))
    return bool(diag['physical_insertion_candidate'] and finite_tree([distance,max_assembly_penetration])
                and max(0.,-distance) <= ENDPOINT_TOLERANCE_M and max_assembly_penetration <= ENDPOINT_TOLERANCE_M)


class PreparedResetView:
    """One-use no-motion reset after explicit physical admission.

    ReferenceRunner.reset runs unchanged, including FD/EMA/reference/noise reset.
    All state belongs to env; the view never resets/teleports qpos or body poses.
    """
    def __init__(self, env, admitted=False):
        if not admitted: raise ValueError('PHYSICAL_ADMISSION_REQUIRED')
        self.env=env
        self.used=False
        self.physics_sink=None
        self.physics_count=0
        self.policy_step=0
        self.physics_timings=[]
        self.contract=None
        self.slip_reference=None
        self.max_arm_limit=0.
        self.max_finger_limit=0.
        self.max_penetration=0.
        self.max_assembly_penetration=0.
        self.max_slip_m=0.
        self.max_slip_rad=0.
        self.max_arm_qvel=0.
        self.arm_motor_saturated=0
        self.arm_motor_total=0

    def __getattr__(self, name): return getattr(self.env,name)

    def reset(self, seed=0, **kwargs):
        if self.used or kwargs: raise ValueError('PREPARED_RESET_IS_ONCE_ONLY_NO_PHYSICAL_CHANGES')
        self.used=True
        return self.env.snapshot()

    def physics_step(self, torque, finger_target=0., record=True):
        if self.physics_sink is None:
            return self.env.physics_step(torque, finger_target=finger_target)
        targets=self.contract.targets() if self.contract is not None else {}
        started=time.perf_counter()
        state=self.env.physics_step(torque, finger_target=finger_target, record=False)
        elapsed=time.perf_counter()-started
        self.physics_timings.append(elapsed)
        self.physics_count+=1
        extra=physical_measurements(self.env, self.slip_reference)
        self.max_arm_limit=max(self.max_arm_limit, extra['arm_joint_limit_violation_rad'])
        self.max_finger_limit=max(self.max_finger_limit, extra['finger_joint_limit_violation_m'])
        self.max_penetration=max(self.max_penetration, extra['all_contact_penetration_m'])
        self.max_assembly_penetration=max(self.max_assembly_penetration, extra['peg_hole_penetration_m'])
        self.max_slip_m=max(self.max_slip_m,extra['peg_hand_slip_m'])
        self.max_slip_rad=max(self.max_slip_rad,extra['peg_hand_slip_rad'])
        self.max_arm_qvel=max(self.max_arm_qvel,float(np.max(np.abs(self.env.data.qvel[self.env.arm_dof_ids]))))
        integrated=np.asarray(state['integration_actuation']['arm_actuator_force_nm'])
        limits=np.max(abs(self.env.model.actuator_forcerange[self.env.arm_actuator_ids]),axis=1)
        self.arm_motor_saturated+=int(np.count_nonzero(np.abs(integrated)>=limits-1e-6))
        self.arm_motor_total+=7
        state['s2a']=extra
        if self.physics_sink is not None:
            line(self.physics_sink,{'policy_step':self.policy_step,'physics_step':self.physics_count,
                                   'physics_wall_s':elapsed,'target':targets,'state':state})
        if not finite_tree(state): raise RuntimeError('NONFINITE_PHYSICAL_MEASUREMENT')
        return state


def independent_peg(env):
    return bool(env.model.jnt_type[env.peg_joint_id] == int(env.mj.mjtJoint.mjJNT_FREE)
                and not np.any(env.model.eq_type == int(env.mj.mjtEq.mjEQ_WELD)))


def joint_limit_violation(env, joint_ids, qpos_ids):
    q=env.data.qpos[qpos_ids]; limits=env.model.jnt_range[joint_ids]
    return float(max(0.,np.max(limits[:,0]-q),np.max(q-limits[:,1])))


def peg_relative(env):
    hand=env.data.xmat[env.model.body('hand').id].reshape(3,3)
    peg=env.data.xmat[env.peg_body_id].reshape(3,3)
    return hand.T@(env.data.xpos[env.peg_body_id]-env.tcp_state()[0]),hand.T@peg


def rotation_distance(a,b):
    return float(math.acos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))


def physical_measurements(env, reference=None):
    contacts=env.contacts()
    assembly=[c for c in contacts if 'peg_collision' in (c['geom1'],c['geom2'])
              and any(g.startswith('hole_') for g in (c['geom1'],c['geom2']))]
    rel,rot=peg_relative(env)
    diag=env.insertion_diagnostics()
    return {'hole_quaternion_wxyz':env.data.xquat[env.hole_body_id].tolist(),
            'finger_generalized_actuator_contribution_n':env.data.qfrc_actuator[env.finger_dof_ids].tolist(),
            'relative_peg_hole':diag,'relative_peg_hand_m':rel.tolist(),'relative_peg_hand_rotation':rot.tolist(),
            'peg_hand_slip_m':0. if reference is None else float(np.linalg.norm(rel-reference[0])),
            'peg_hand_slip_rad':0. if reference is None else rotation_distance(rot,reference[1]),
            'arm_joint_limit_violation_rad':joint_limit_violation(env,env.arm_joint_ids,env.arm_qpos_ids),
            'finger_joint_limit_violation_m':joint_limit_violation(env,env.finger_joint_ids,env.finger_qpos_ids),
            'all_contact_penetration_m':max((max(0.,-c['distance_m']) for c in contacts),default=0.),
            'peg_hole_penetration_m':max((max(0.,-c['distance_m']) for c in assembly),default=0.)}


def reset_draw(seed):
    rng=np.random.default_rng(seed)
    offset=rng.uniform(-1.,1.,3)*[.02,.02,.01]
    yaw=float(rng.uniform(-.785,.785))
    held=rng.uniform(-1.,1.,3)*[.003,0.,.003]
    return rng,{'seed':seed,'hole_root_m':[.6,0.,.05],'hole_quaternion_wxyz':[1.,0.,0.,0.],
                'tcp_position_offset_m':offset.tolist(),'tcp_target_m':(np.array([.6,0.,.122])+offset).tolist(),
                'tcp_target_quaternion_wxyz':quat_from_euler(3.1416,0.,yaw).tolist(),
                'tcp_yaw_rad':yaw,'held_noise_local_m':held.tolist()}


def solve_task_reset(env, draw):
    q=SOURCE_RESET_Q.copy()
    lower,upper=env.model.jnt_range[env.arm_joint_ids].T
    error=np.zeros(6)
    for iteration in range(300):
        env.data.qpos[env.arm_qpos_ids]=q
        env.mj.mj_forward(env.model,env.data)
        p,r=env.tcp_state()
        error=np.r_[np.asarray(draw['tcp_target_m'])-p,
                    axis_angle(quat_mul(draw['tcp_target_quaternion_wxyz'],quat_conjugate(r)))]
        if np.linalg.norm(error)<1e-5: break
        j=env.tcp_jacobian()
        q=np.clip(q+.5*j.T@np.linalg.solve(j@j.T+1e-4*np.eye(6),error),lower,upper)
    return {'accepted':bool(np.linalg.norm(error)<1e-5),'qpos':q.tolist(), 'iterations':iteration+1,
            'position_error_m':float(np.linalg.norm(error[:3])),'orientation_error_rad':float(np.linalg.norm(error[3:]))}


def finger_forces(env):
    forces={'left_finger':0.,'right_finger':0.}; interference=set()
    for c in env.contacts():
        if 'peg_collision' not in (c['geom1'],c['geom2']): continue
        other=c['geom2'] if c['geom1']=='peg_collision' else c['geom1']
        body=env.model.body(int(env.model.geom_bodyid[env.model.geom(other).id])).name
        # Panda pad geometries, not a hand/palm collision, are grasp evidence.
        if body in forces and any(other.endswith('_'+str(k)) for k in range(3,8)):
            forces[body]+=max(0.,c['normal_force_n'])
        else: interference.add(body)
    return forces, sorted(interference)


def prepare_physical_episode(env, seed, directory):
    rng,draw=reset_draw(seed)
    write(directory/'reset_draw.json',draw)
    ik=solve_task_reset(env,draw)
    write(directory/'ik.json',ik)
    if not ik['accepted']: return {'admission':'FAIL','failure_kind':'RESET_IK_FAILURE','ik':ik,'draw':draw}
    env.data.qpos[env.arm_qpos_ids]=ik['qpos'];env.mj.mj_forward(env.model,env.data)
    tcp,_=env.tcp_state(); hand=env.data.xmat[env.model.body('hand').id].reshape(3,3).copy()
    flipped=hand@np.diag([-1.,1.,-1.]); quat=np.zeros(4)
    env.mj.mju_mat2Quat(quat,flipped.reshape(-1))
    peg_root=tcp-flipped[:,2]*(.05-.017608)+flipped@draw['held_noise_local_m']
    env.reset(seed=seed,arm_qpos=ik['qpos'],peg_pos=peg_root,peg_quat=quat,
              hole_pos=draw['hole_root_m'],hole_quat=draw['hole_quaternion_wxyz'],finger_opening=.007986/2*1.25)
    env.rng=rng  # Continue the registered local draw stream; no duplicate noise draw.
    initial=env.snapshot(); relatives=[]; rotations=[]; positives=[]; interferences=set(); peak=0.; finite=True
    with (directory/'closure.jsonl').open('x') as stream:
        for step in range(30):
            state=env.physics_step(np.zeros(7),finger_target=0.,record=False)
            rel,rot=peg_relative(env);forces,bodies=finger_forces(env)
            bodies += ['non_grasp_contact:'+str(c['geom1'])+':'+str(c['geom2']) for c in state['contacts']
                       if 'peg_collision' not in (c['geom1'],c['geom2']) and c['distance_m'] < -1e-6]
            relatives.append(rel);rotations.append(rot);positives.append(all(v>1e-6 for v in forces.values()))
            if step>=20: interferences.update(bodies)
            peak=max(peak,max((max(0.,-c['distance_m']) for c in state['contacts']),default=0.))
            finite &= finite_tree(state)
            line(stream,{'closure_step':step+1,'state':state,'pad_force_n':forces,'interference_bodies':bodies,
                         'relative_peg_hand_m':rel,'relative_peg_hand_rotation':rot})
    diagnostic=env.insertion_diagnostics()
    metrics={'bilateral_last10':all(positives[-10:]),'finite':bool(finite), 'independent_peg':independent_peg(env),
             'starts_above_hole':bool(diagnostic['entirely_above_opening'] and env.tcp_state()[0][2]>.075),
             'stability_m':max(float(np.linalg.norm(x-relatives[-1])) for x in relatives[-10:]),
             'angular_stability_rad':max(rotation_distance(x,rotations[-1]) for x in rotations[-10:]),
             'q_limit_violation':max(joint_limit_violation(env,env.arm_joint_ids,env.arm_qpos_ids),
                                     joint_limit_violation(env,env.finger_joint_ids,env.finger_qpos_ids)),
             'interference_bodies':sorted(interferences),'maximum_closure_all_contact_penetration_m':peak,
             'final_pad_forces_n':finger_forces(env)[0],'initial':initial,'final':env.snapshot(),
             'draw':draw,'ik':ik,'closure_steps':30,'diagnostic':diagnostic}
    errors=admission_errors(metrics)
    metrics.update({'admission':'FAIL' if errors else 'PASS','admission_errors':errors,
                    'failure_kind':'PHYSICAL_GRASP_INITIALIZATION_FAILURE' if errors else None})
    return metrics


def stats(values):
    if not values: return None
    return {'mean':float(np.mean(values)), 'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95)),'p99':float(np.percentile(values,99)), 'maximum':float(max(values))}


def run_episode(policy, seed, directory, budget_start):
    directory.mkdir(exist_ok=False)
    write(directory/'attempt.json',{'seed':seed,'attempt':1,'retries_allowed':0,'started_utc':datetime.now(timezone.utc).isoformat()})
    env=AdaptedPegInsertEnv(EnvironmentConfig())
    try:
        reset=prepare_physical_episode(env,seed,directory)
    except Exception as exc:
        result={'seed':seed,'reset_admitted':False,'policy_steps_completed':0,'policy_inferences':0,
                'physics_steps_completed':0,'reset_physics_steps':round(float(env.data.time)/PHYSICS_DT),
                'termination_reason':'RESET_RUNTIME_FAILURE','error':str(exc),'traceback':traceback.format_exc()}
        write(directory/'result.json',result)
        return result
    write(directory/'reset_admission.json',reset)
    result={'seed':seed,'reset_admitted':reset['admission']=='PASS','policy_steps_completed':0,
            'policy_inferences':0,'policy_inference_requests':0,'physics_steps_completed':0,'reset_physics_steps':round(float(env.data.time)/PHYSICS_DT),
            'termination_reason':reset.get('failure_kind')}
    if not result['reset_admitted']:
        write(directory/'result.json',result);return result
    view=PreparedResetView(env,admitted=True);runner=ReferenceRunner(view)
    try:
        before=env.data.qpos.copy();initial=runner.reset(seed=seed)
        if not np.array_equal(before,env.data.qpos): raise RuntimeError('PREPARED_RESET_MOVED_PHYSICAL_STATE')
        view.contract=runner.contract;view.slip_reference=peg_relative(env)
        lstm=policy.reset(seed)
        if any(np.count_nonzero(np.asarray(lstm[k])) for k in ('h','c')): raise RuntimeError('NONZERO_LSTM_RESET')
        write(directory/'interface_reset.json',{'initial':initial,'lstm':lstm,'qpos_unchanged':True})
    except Exception as exc:
        result.update({'termination_reason':'INTERFACE_RESET_FAILURE','error':str(exc),'traceback':traceback.format_exc()})
        write(directory/'result.json',result)
        return result
    trace={k:[] for k in ('observation','normalized','mu','clipped','ema','h','c')}
    latencies=[];ipc_latencies=[];step_times=[];command_times=[];ever=False;first=None;out=None;fault=None
    wall=time.perf_counter();cpu=time.process_time();saturated=0
    try:
        with (directory/'physics.jsonl').open('x') as physics_log,(directory/'policy.jsonl').open('x') as policy_log, (directory/'inference.jsonl').open('x') as inference_log:
            view.physics_sink=physics_log
            for step in range(1,MAX_POLICY_STEPS+1):
                if time.perf_counter()-budget_start>MAX_RUNTIME_S: raise TimeoutError('EXECUTION_BUDGET_EXCEEDED')
                step_start=time.perf_counter();obs=runner.contract.observation().copy()
                result['policy_inference_requests']+=1
                line(inference_log,{'event':'request','policy_step':step,'observation':obs})
                infer_start=time.perf_counter();actor=policy.infer(obs.tolist());result['policy_inferences']+=1
                line(inference_log,{'event':'response','policy_step':step,'actor':actor})
                ipc_latencies.append(time.perf_counter()-infer_start);validate_policy_result(actor)
                if not np.array_equal(np.asarray(actor['observation'],dtype=np.float32),obs): raise ValueError('OBSERVATION_TRANSFER_CHANGED')
                view.policy_step=step;command_times.append(time.perf_counter())
                cache_before=runner.contract.action_cache.copy()
                out=runner.step_reference_action(np.asarray(actor['clipped'],dtype=np.float32))
                expected=.8*cache_before+.2*np.asarray(actor['clipped'],dtype=np.float32)
                if not np.allclose(out['ema_cache'],expected,atol=1e-7,rtol=1e-7): raise RuntimeError('EMA_TRANSFER_MISMATCH')
                if not out['state_is_pre_reset']: raise RuntimeError('POST_RESET_OUTCOME_REJECTED')
                result['policy_steps_completed']=step
                ever |= bool(out['official_geometric_success'])
                if ever and first is None:first=step
                saturated+=int(np.count_nonzero(np.abs(actor['mu'])>=1))
                for k in ('observation','normalized','mu','clipped','h','c'):trace[k].append(actor[k])
                trace['ema'].append(out['ema_cache'].tolist())
                latencies.append(actor['latency_seconds']);step_times.append(time.perf_counter()-step_start)
                line(policy_log,{'policy_step':step,'observation':actor['observation'],'normalized':actor['normalized'],
                     'mu':actor['mu'],'clipped':actor['clipped'],'ema':out['ema_cache'],
                     'lstm_reset_event':step==1,'lstm_state_in_npz_row':step-1,
                     'official_geometric_success':out['official_geometric_success'],
                     'physical_diagnostic':out['physical_insertion_diagnostic'],
                     'simulation_time_s':out['episode_simulation_time_s'],'state_is_pre_reset':out['state_is_pre_reset'],
                     'inference_latency_s':actor['latency_seconds'],'ipc_latency_s':ipc_latencies[-1],
                     'wall_step_s':step_times[-1],'control_substeps':out['substeps']})
    except Exception as exc:
        fault={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
               'runner_last_fault':runner.last_fault,'physical_state':env.snapshot()}
        write(directory/'runtime_failure.json',fault)
    elapsed=time.perf_counter()-wall
    np.savez_compressed(directory/'actor_trace.npz',**{k:np.asarray(v,dtype=np.float32) for k,v in trace.items()})
    actual_physics_steps=round((float(env.data.time)-runner.episode_start_time)/PHYSICS_DT)
    try:
        info=policy.info()
    except Exception as exc:
        info={'rms_unchanged':False,'eval_mode':False,'error':str(exc)}
        fault=fault or {'type':type(exc).__name__,'message':str(exc)}
    unchanged=sha(CHECKPOINT)==CHECKPOINT_SHA
    gates={'checkpoint_identity':unchanged,'rms_unchanged':info['rms_unchanged'],'model_eval':info['eval_mode'],
           'physical_finite':finite_tree(env.snapshot()),'independent_peg':independent_peg(env),
           'interfaces_valid':fault is None,'complete_149':result['policy_steps_completed']==149,
           'physical_steps_1192':actual_physics_steps==1192,'all_physics_steps_recorded':actual_physics_steps==view.physics_count,
           'arm_joint_limits_within_1e_6_rad':view.max_arm_limit<=1e-6,
           'finger_joint_limits_within_1e_6_m':view.max_finger_limit<=1e-6,'reset_not_hidden':True}
    diag=env.insertion_diagnostics()
    result.update({'physics_steps_completed':actual_physics_steps,'physics_steps_recorded':view.physics_count,
        'technical_gates':gates,'technical_pass':all(gates.values()),
        'termination_reason':'runtime_failure' if fault else 'timeout_149_policy_steps','fault':fault,
        'terminal_success_scope':'last completed pre-reset policy step; final_physical_diagnostic is actual final state',
        'ever_success':ever,'terminal_success':bool(out['official_geometric_success']) if out else False,
        'first_success_policy_step':first,'first_success_simulation_time_s':None if first is None else first*8/120,
        'physical_insertion_candidate':diag['physical_insertion_candidate'],
        'contact_quality_validated_success':contact_quality(diag,view.max_assembly_penetration),
        'final_physical_diagnostic':diag,'maximum_episode_all_contact_penetration_m':view.max_penetration,
        'maximum_episode_peg_hole_penetration_m':view.max_assembly_penetration,
        'maximum_peg_hand_slip_m':view.max_slip_m,'maximum_peg_hand_slip_rad':view.max_slip_rad,
        'maximum_arm_joint_limit_violation_rad':view.max_arm_limit,'maximum_finger_joint_limit_violation_m':view.max_finger_limit,
        'maximum_arm_qvel_rad_s':view.max_arm_qvel,
        'action_saturation_rate':saturated/(6*len(latencies)) if latencies else None,
        'arm_motor_saturation_fraction':view.arm_motor_saturated/view.arm_motor_total if view.arm_motor_total else None,
        'simulation_time_s':view.physics_count/120,'wall_clock_s':elapsed,'dynamics_cpu_s':time.process_time()-cpu,
        'inference_latency_s':stats(latencies),'ipc_inference_latency_s':stats(ipc_latencies),'physics_step_wall_s':stats(view.physics_timings),
        'policy_step_wall_s':stats(step_times),'effective_wall_command_hz':len(latencies)/elapsed,
        'command_submit_hz':None if len(command_times)<2 else (len(command_times)-1)/(command_times[-1]-command_times[0]),
        'real_time_factor':(view.physics_count/120)/elapsed,'policy_info_after':info})
    write(directory/'result.json',result)
    return result


def main():
    from scripts.g4s2a_inference import InferenceClient
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-episodes',type=int,default=3,choices=(1,2,3))
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();episodes=[];failure=None;actor_receipt=None
    xml=model_xml(EnvironmentConfig());verify_identity(CHECKPOINT,xml)
    import mujoco
    if mujoco.__version__!='3.5.0':raise RuntimeError('MUJOCO_VERSION_MISMATCH')
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/g4s2a_smoke.py',ROOT/'scripts/g4s2a_inference.py',
             ROOT/'scripts/g4s1b_env.py',ROOT/'scripts/g4s1b_contract.py',ROOT/'scripts/g4i1_policy.py']}
    protocol={'task':'G4-S2A','start_commit':START,'generated_utc':datetime.now(timezone.utc).isoformat(),
      'seeds':list(SEEDS[:args.max_episodes]),'max_complete_episodes':3,'max_policy_steps':149,'max_execution_wall_s':1200,
      'checkpoint_path':str(CHECKPOINT),'checkpoint_sha256':CHECKPOINT_SHA,'xml_sha256':XML_SHA,
      'mujoco_version':mujoco.__version__,'menagerie_commit':MENAGERIE_COMMIT,'configuration':asdict(EnvironmentConfig()),
      'reset':'one IK solve from original reset q; original TCP xyz/yaw + held noise ranges; fixed hole/support pose; 30 zero-arm-torque physical finger closure steps',
      'reset_rng':'NumPy default_rng(seed); TCP xyz, yaw, held xyz, then observation Gaussian; not paired PhysX',
      'admission':'last10 bilateral pad force>1e-6N; translation stability<=1mm; rotation<=.05rad; limits<=1e-6 numerical tolerance; no peg nonpad interference; entire peg initially above opening; free peg/no weld',
      'continuation_joint_limits':'whole-episode max arm/finger violation <=1e-6 rad/m; larger excursions stop additional seeds',
      'contact_quality':'S1B independent candidate + endpoint assembly penetration<=10um AND stricter entire episode peg/hole penetration<=10um; report gripper/all-contact peaks separately',
      'source_files_sha256':sources,'automatic_retry':False,'training':False,'s2b':False}
    write(output/'preregistered_protocol.json',protocol)
    client=None
    try:
        client=InferenceClient(ACTOR_PYTHON,CHECKPOINT,repo=ROOT,stderr_path=output/'actor_stderr.log')
        write(output/'actor_initial_info.json',client.info())
        for seed in SEEDS[:args.max_episodes]:
            if time.perf_counter()-started>MAX_RUNTIME_S:raise TimeoutError('EXECUTION_BUDGET_EXCEEDED')
            result=run_episode(client,seed,output/f'seed_{seed}',started);episodes.append(result)
            print(json.dumps({'seed':seed,'reset_admitted':result['reset_admitted'],
                 'steps':result['policy_steps_completed'],'technical_pass':result.get('technical_pass',False)}),flush=True)
            if not result.get('technical_pass',False):break
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    finally:
        if client is not None:
            try:actor_receipt=client.close()
            except Exception as exc:actor_receipt={'close_error':str(exc)}
    valid=sum(bool(x.get('technical_pass')) for x in episodes)
    actor_exit_ok=bool(actor_receipt and actor_receipt.get('child_returncode')==0 and actor_receipt.get('healthy'))
    status='G4_S2A_PASS' if valid and actor_exit_ok and failure is None and all(x.get('technical_pass') for x in episodes) else ('G4_S2A_PARTIAL' if any(x['policy_inferences'] for x in episodes) else 'G4_S2A_BLOCKED')
    summary={'task':'G4-S2A','status':status,'protocol':'ADAPTED_ZERO_SHOT_SIM2SIM','episodes':episodes,
      'attempted_seed_denominator':len(list(output.glob('seed_*/attempt.json'))),'completed_valid_episode_denominator':valid,
      'reset_failures':sum(not x['reset_admitted'] for x in episodes),'failure':failure,'actor_process_receipt':actor_receipt,
      'policy_inferences':sum(x['policy_inferences'] for x in episodes),'physics_steps':sum(x['physics_steps_completed'] for x in episodes),
      'reset_physics_steps':sum(x['reset_physics_steps'] for x in episodes),'actor_exit_ok':actor_exit_ok,
      'wall_s':time.perf_counter()-started,'dynamics_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'checkpoint_unchanged':sha(CHECKPOINT)==CHECKPOINT_SHA,'learning_status':'PENDING','stop_after':'G4-S2A'}
    write(output/'summary.json',summary)
    files={str(p):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'artifact_manifest.json',files)
    print(json.dumps({'status':status,'attempted':summary['attempted_seed_denominator'],'complete':valid,'failure':failure}),flush=True)
    return 0 if status=='G4_S2A_PASS' else 2

if __name__=='__main__':raise SystemExit(main())
