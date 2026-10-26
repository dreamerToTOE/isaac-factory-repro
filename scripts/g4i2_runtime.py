"""Optional telemetry adapter for the existing G4-I1 runtime, never a policy.

The unchanged environment owns all observations, EMA, targets, controller calls,
physics and success. Detached diagnostics are serialized after command submission.
Host command timestamps denote wrapper admission, not actuator execution.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from types import MethodType

from scripts.g4i2_performance import StepProfiler, DeadlineGuard, command_interval_summary


class Telemetry:
    def __init__(self, args, env, policy, state, observation):
        import torch
        from isaaclab_tasks.direct.factory import factory_control as fc
        from scripts.g4i2_controller_probe import TorqueProbe
        self.torch, self.args, self.env, self.policy = torch, args, env, policy
        self.fc, self.original_torque = fc, fc.compute_dof_torque
        self.torque = TorqueProbe(self.original_torque) if args.diagnostics=='full' else None
        if self.torque is not None: fc.compute_dof_torque = self.torque
        self.probe = StepProfiler(args.profiling, device=args.device)
        # Registered log-only deadlines preserve the scientific action sequence.
        # A 60s software freshness ceiling admits slow simulated resets; no ROS clock claim.
        self.deadline = DeadlineGuard(mode='log_only', max_observation_age_ms=60000.)
        self.output = args.output
        self.stream = (self.output/'i2_trace.jsonl').open('x')
        self.step_id = 0; self.command_stamps = []; self.episode_rows = []
        self.initials = {}; self.target_context = None; self.active = False
        self.observation_started_ns = time.perf_counter_ns()
        self.observation_available_ns = self.observation_started_ns
        self.original_observe, self.original_physics = env._get_observations, env.sim.step
        original_observe = self.original_observe
        def observe(instance):
            self.observation_started_ns = time.perf_counter_ns()
            with self.probe.section('observation_construction', cuda=True):
                result = original_observe()
            self.observation_available_ns = time.perf_counter_ns()
            return result
        env._get_observations = MethodType(observe, env)
        def physics(*a, **kw):
            with self.probe.section('physics_step', cuda=True):
                return self.original_physics(*a, **kw)
        env.sim.step = physics
        self.save_initials(range(env.num_envs), observation, state, initial=True)
        torch.save({'cpu': torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state(args.device)}, self.output/'initial_rng.pt')

    def plain(self, x):
        if self.torch.is_tensor(x): return x.detach().cpu().tolist()
        return x

    def emit(self, row):
        self.stream.write(json.dumps(row, allow_nan=False)+'\n')

    def snapshot(self):
        e = self.env
        fields = ['joint_pos', 'joint_vel', 'fingertip_midpoint_pos', 'fingertip_midpoint_quat',
                  'ee_linvel_fd','ee_angvel_fd', 'held_pos','held_quat','held_base_pos','target_held_base_pos',
                  'fixed_pos','fixed_quat','fixed_pos_obs_frame','fixed_pos_action_frame','init_fixed_pos_obs_noise','actions',
                  'task_prop_gains','task_deriv_gains','pos_threshold','rot_threshold']
        result = {k:self.plain(getattr(e,k)) for k in fields}
        # Complete simulator generalized/root state complements task randomized frames.
        for name, obj in [('robot',e._robot),('held',e._held_asset),('fixed',e._fixed_asset)]:
            for attr in ['root_state_w','joint_pos','joint_vel']:
                value = getattr(obj.data, attr, None)
                if value is not None: result[name+'__'+attr] = self.plain(value)
        return result

    def save_initials(self, ids, obs, state, initial=False):
        snapshot = self.snapshot()
        for env_id in ids:
            episode_id = 0 if initial else self.initials[env_id]['episode_id']+1
            record = {'seed':self.args.seed,'env_id':env_id,'episode_id':episode_id,
                      'policy_step':self.step_id,'observation':self.plain(obs[env_id]),
                      'randomized_state':{k:(v[env_id] if isinstance(v,list) and len(v)==self.env.num_envs else v) for k,v in snapshot.items()},
                      'lstm_h':self.plain(state[0][:,env_id,:]),'lstm_c':self.plain(state[1][:,env_id,:]),
                      'provenance':'actual_simulator_post_reset_observation; no synthetic injection'}
            self.initials[env_id]=record
            self.emit({'kind':'initial_state',**record})

    def begin(self, index):
        self.step_id=index; self.warmup=index<=self.args.warmup
        self.first_effort_target_ns=None
        self.buffer=[]; self.start=time.perf_counter_ns(); self.active=True
        self.input_observation_started_ns=self.observation_started_ns
        self.context=self.probe.step(index,warmup=self.warmup); self.context.__enter__()
        self.row_start=len(self.probe.rows)
        # The policy consumes GPU-resident observations: no H2D copy is performed.
        with self.probe.section('cpu_gpu_transfer_h2d',cuda=False): pass

    def target_begin(self):
        self.target_context=self.probe.section('controller_target',cuda=True)
        self.target_context.__enter__()

    def target_end(self, env):
        if self.target_context is not None:
            self.target_context.__exit__(None,None,None);self.target_context=None
        # No tensor conversions or synchronization before original controller call.

    def controller(self, env):
        if self.first_effort_target_ns is None:
            self.first_effort_target_ns=time.perf_counter_ns()
            self.command_stamps.append(self.first_effort_target_ns)
            self.full_obs_to_command_ms=(self.first_effort_target_ns-self.input_observation_started_ns)/1e6
        if self.torque is None: return
        last=self.torque.last
        requested=last.requested_torque[:,:7]
        software=last.postclip_torque[:,:7]
        if not self.torch.equal(software, env.joint_torque[:,:7]):
            raise RuntimeError('controller probe changed software torque')
        predicted=env.fingertip_midpoint_pos + env.actions[:,:3]*env.pos_threshold
        displacement=predicted-env.fixed_pos_action_frame
        active=(displacement < -env.cfg.ctrl.pos_action_bounds[0]) | (displacement > env.cfg.ctrl.pos_action_bounds[1])
        configured=env._robot.data.joint_effort_limits[:,:7]
        record={'kind':'controller','policy_step':self.step_id,
                'substep':sum(r['kind']=='controller' for r in self.buffer)+1,
                'target_position':env.ctrl_target_fingertip_midpoint_pos.detach().clone(),
                'target_quaternion':env.ctrl_target_fingertip_midpoint_quat.detach().clone(),
                'requested_torque':requested,'software_torque':software,
                'effort_limits':configured.detach().clone(),
                'task_space_pd_output':last.task_wrench,
                'position_constraint_active':active,
                'orientation_constraint_active':self.torch.tensor([[True,True,False]],device=env.device).expand(env.num_envs,-1),
                'applied_sim_effort':'UNKNOWN; target/effort limit do not measure applied physical torque'}
        self.buffer.append(record)

    def action(self, step, cache):
        if self.args.diagnostics=='minimal': return
        ema=.2*step.clipped_action+.8*cache
        self.buffer.append({'kind':'action','policy_step':self.step_id,
             'raw_mu':step.mu,'clipped':step.clipped_action,'ema':ema,
             'observation':step.observation,'normalized':step.normalized,
             'physical_position_increment_m':ema[:,:3]*self.env.pos_threshold,
             'physical_rotation_increment_rad':ema[:,3:]*self.env.rot_threshold})

    def submit(self, original):
        def admitted(actions):
            stamp=time.perf_counter_ns()
            decision=self.deadline.check(observation_ns=self.observation_started_ns,
                      inference_started_ns=self.start,now_ns=stamp)
            if not decision.allowed: raise RuntimeError('I2 command rejected: '+decision.reason)
            self.obs_to_command_ms=(stamp-self.observation_started_ns)/1e6
            self.available_to_command_ms=(stamp-self.observation_available_ns)/1e6
            self.admission=decision.to_dict()
            return original(actions)
        return admitted

    def episodes(self, finalized, cap, done, obs, state):
        self.buffer.append({'kind':'success_state','policy_step':self.step_id,'capture':cap})
        snapshots=cap['geometric_snapshots']
        for episode in finalized:
            d=episode.to_dict(); idx=episode.env_id
            d.update(seed=self.args.seed,evaluation_seed=self.args.seed,first_success_time_seconds=None if episode.first_success_step is None else episode.first_success_step*self.env.step_dt,
                     initial_randomized_state=self.initials[idx],initial_randomized_configuration=self.initials[idx],
                     pre_reset_terminal_state={k:(v[idx] if isinstance(v,list) and len(v)==self.env.num_envs else v) for k,v in snapshots.items()},
                     termination_reason=episode.terminal_kind,
                     failure_class='none' if episode.terminal_success else ('transient_success_lost' if episode.ever_success else 'no_success_before_timeout'),
                     trace={'path':str(self.output/'i2_trace.jsonl'),'env_id':idx,'episode_id':episode.episode_id,'first_policy_step':self.step_id-episode.steps+1,'last_policy_step':self.step_id},
                     statistics_scope='all per-env raw action/controller samples and per-batch raw timing retained in trace')
            self.episode_rows.append(d);self.emit({'kind':'episode',**d})
        self.done_ids=done.nonzero(as_tuple=False).flatten().tolist()

    def end(self, obs, state):
        with self.probe.section('cpu_gpu_transfer_d2h',cuda=True):
            converted=[{k:self.plain(v) for k,v in record.items()} for record in self.buffer]
        # Deferred sync resolves all events before statistics. Diagnostics remain measured overhead.
        self.context.__exit__(None,None,None);self.active=False
        for row in converted: self.emit({**row,'warmup':self.warmup})
        if self.done_ids: self.save_initials(self.done_ids,obs,state)
        rows=self.probe.rows[self.row_start:]
        timings={}
        for row in rows:
            timings[row['stage']]=timings.get(row['stage'],0.)+row['wall_ns']/1e6
        timings['observation_to_command']=self.full_obs_to_command_ms
        timings['wrapper_admission']=self.obs_to_command_ms
        timings['available_observation_to_command']=self.available_to_command_ms
        # Physics and target scopes have 8+ samples per decision, totals are explicitly labelled.
        self.emit({'kind':'timing','policy_step':self.step_id,'warmup':self.warmup,
                   'timings_ms':timings,'wall_command_monotonic_ns':self.command_stamps[-1],
                   'control_loop_ms':((self.command_stamps[-1]-self.command_stamps[-2])/1e6 if len(self.command_stamps)>1 else (time.perf_counter_ns()-self.start)/1e6),
                   'sim_dt_s':self.env.step_dt,'admission':self.admission,
                   'event_sections':rows,'step_timing':self.probe.steps[-1] if self.probe.steps else None})
        self.emit({'kind':'pre_reset_state','policy_step':self.step_id, 'observation_next':self.plain(obs)})
        self.stream.flush()

    def finish(self, tracker, result):
        from scripts.g4i2_analysis import summarize_timings,summarize_actions,summarize_controller,summarize_episodes,write_episode_csv
        self.stream.flush()
        records=[json.loads(x) for x in (self.output/'i2_trace.jsonl').read_text().splitlines()]
        timing=[x for x in records if x['kind']=='timing']
        summaries={'timing':summarize_timings(timing),'actions':summarize_actions([x for x in records if x['kind']=='action']),
                   'controller':summarize_controller([x for x in records if x['kind']=='controller']),
                   'episodes':summarize_episodes(self.episode_rows),'event_profile':self.probe.summarize(),
                   'command_intervals':command_interval_summary(self.command_stamps,warmup_commands=self.args.warmup),
                   'admission':self.deadline.summarize()}
        for name,data in [('summary',summaries),('episodes',self.episode_rows)]:
            (self.output/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        write_episode_csv(self.episode_rows,self.output/'episodes.csv')
        failures=[x for x in self.episode_rows if not x['terminal_success']]
        (self.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
        return {'profiling':self.args.profiling,'warmup':self.args.warmup,'episodes':len(self.episode_rows),
                'failures':len(failures),'scope':'all diagnostics measured; events-off still has trace overhead',
                'timing_finite':True,'summary_path':str(self.output/'summary.json')}

    def close(self):
        if self.active:
            try:
                for row in self.buffer: self.emit({**{k:self.plain(v) for k,v in row.items()},'aborted':True})
                self.emit({'kind':'abort','policy_step':self.step_id,'commands_submitted':len(self.command_stamps)})
            except Exception: pass
            self.context.__exit__(RuntimeError,RuntimeError('runtime aborted'),None)
        self.env._get_observations=self.original_observe
        self.env.sim.step=self.original_physics
        self.fc.compute_dof_torque=self.original_torque
        self.stream.close()
