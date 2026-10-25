# G3B read-only source and scientific audit

Audit date: 2026-10-25, Asia/Seoul. Auditor did not run simulation, change the repository, load checkpoints in Torch, install anything, or touch G3V work. Initial machine-readable evidence: `work/g3b_source_audit_initial.json` beside this file.

## Identity and preservation

Repository HEAD is expected G3A `668d31f412404542bc5d3ccd4822512577663115`. Tracked baseline was clean at first inspection. Five pre-existing untracked G3V files were listed and SHA256 recorded separately; they must remain outside this change. All 316 tracked G2/G3A evidence files were byte-compared with `git show HEAD:path`: no differences. Upstream Isaac Lab HEAD is `90b79bb2d44feb8d833f260f2bf37da3487180ba`, working tree clean. All 17 prior audited upstream/RL-Games/runtime source files match their G2 recorded hashes.

The actual final checkpoint was read through streaming SHA256, without deserialization: 212,889,225 bytes, SHA256 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`. Path is the G2 selected-checkpoint record's original local path, basename `last_Factory_ep_200_rew__350.4961_.pth`.

Original epoch200 metrics independently yield:

| Seed | Ever | Terminal | Failures (env, episode) |
| --- | --- | --- | --- |
| 1001 | 95/96 | 95/96 | (11,0) |
| 1002 | 96/96 | 96/96 | none |
| 1003 | 89/96 | 89/96 | (4,0), (12,0), (25,0), (29,0), (5,1), (1,2), (5,2) |

Every original episode is 149 steps. Each of the eight failures has `ever_success=false`, `terminal_success=false`, `first_success_step=null`; the seed files each retain only 12 selected successful geometry snapshots, not complete failed trajectories. No exact historical replay can therefore be proven by matching IDs or outcomes alone. Original primary aggregate remains 280/288. Source metrics SHA256 are in the JSON audit.

## Coordinates, predicate, and timing

Pinned `factory_env.py` lines194–203 obtains object and end-effector positions as world positions minus `scene.env_origins`, while quaternions and velocities remain world-axis oriented. These are translated environment-local positions, not rotations into the hole frame. Quaternions use scalar-first **wxyz**; identity is `[1,0,0,0]`.

For PegInsert, held-base local offset is zero and target fixed-success local offset is zero (lines91–107 and133–140). Thus held-base point equals peg root and target point equals hole root. `held_base_quat` equals held root quaternion and target quaternion equals fixed root quaternion. Do not substitute noisy controller/observation fixed frames for these true geometry tensors.

Official `_get_curr_successes` (lines431–455) uses **world-axis** `||held_xy-target_xy|| < 0.0025m` and `held_z-target_z < height*threshold`. PegInsert socket height25mm and success threshold0.04 give a strict **z gap <1mm**. This is one-sided, with no lower bound. PegInsert `check_rot=False`; it has no orientation criterion. Engagement uses same XY gate and 0.9 height fraction, so z gap<22.5mm. These exact official booleans should be observed rather than replaced by a new geometric implementation.

Additional useful definitions are hole-local axial gap `dot(held_base-target, hole_positive_Z_world)`, hole-local transverse separation, and insertion depth proxy `25mm - local_axial_gap`. Label sign, reference origins, and whether values are world z or hole-local projection. The nominal plausible depth proxy range0–25mm corresponds to peg base between hole top and bottom under ideal geometry; it is not independently measured volume overlap or physical validity. A negative z/local gap alone does not prove interpenetration. Upper peg height50mm, nominal peg diameter7.986mm, nominal hole diameter8.1mm imply nominal radial clearance57µm; these config dimensions do not certify detailed USD/collision mesh clearance or assembly quality.

Axis tilt must rotate local **+Z** using each body's quaternion. Directed tilt is acos(clamped axis dot) in0–180degrees; unoriented line tilt may use abs(dot) but must be labelled separately, since it treats upside-down axes as parallel. Full relative quaternion angle includes yaw around a cylindrical peg; it is not the same as relevant axial misalignment. Axis errors are secondary diagnostics, not official criteria. Do not infer a causal orientation failure merely from nonzero tilt without measured/statistically justified evidence.

`DirectRLEnv.step` advances decimated physics, increments episode counter, calls `_get_dones`, then `_get_rewards`, then resets done environments, then computes returned observation. Factory `_get_dones` refreshes intermediate geometry; it marks done at `episode_length_buf >= max_episode_length-1`. Duration10s and step dt1/15s lead max_episode_length150 and149 actual pre-reset steps. Capture inside original success-call hook during reward therefore observes refreshed pre-reset state and terminal `reset_buf`, while readings after `env.step` at done are already reset state. Record terminal geometry there. First-success steps are1-based; duration at successful sampled policy states is success-step-count/15 seconds, not substep contact/continuous dwell proof.

`evaluate_checkpoint.py` uses checkpoint normalization, deterministic policy mean, Torch inference, action clipping and explicit recurrent reset for completed envs. It verifies one success capture per policy step, synchronous dones, terminal extras consistency and returned reset counters0. Extending capture should not call policy/RNG, mutate tensors, run physics, alter success calls, alter reset timing, or add rendering to numerical diagnosis. A GPU tensor buffer and one batched conversion after rollout avoid extra synchronization. Capture the action corresponding to the current transition, never the next action or post-reset observations.

## Contact and controller distinctions

Factory has `activate_contact_sensors=True` in existing asset spawn configs, but no configured `ContactSensor` instance in this scene. Thus there is no already-retained dedicated peg↔hole contact-force tensor. Adding a sensor changes the scene/interface and must not be silently conflated with an existing measurement.

Existing `ArticulationData.body_incoming_joint_wrench_b` is a supported read-only property (articulation_data.py641–654), backed by PhysX `get_link_incoming_joint_force`. Its documentation says parent→child **joint reaction wrench**, parent-body frame, shape[num_instances,num_bodies,6], including root. It is a measured simulation constraint/reaction signal, **not isolated peg–hole contact force**; actuator/inertial/gravity/joint effects and frame semantics prevent that interpretation. Source availability is confirmed, runtime availability/meaningfulness must be checked before claiming any real measurements. Missing or unsupported data must remain missing rather than zero. No extra simulation step or physical setting is necessary for supported read-only access.

`FactoryEnv.applied_wrench` comes from `factory_control.compute_dof_torque`: PD task-space commanded wrench; joint torques are then mapped by Jacobian and nullspace control. It is **controller command**, never measured contact force. Finite differences of geometry, low motion, action magnitudes and nominal overlap/depth are **geometric/motion proxies**. None alone proves contact-related obstruction. Category D should stay unassigned or insufficient evidence unless adequate independent contact evidence is actually available.

## Classification guardrails

A and B may describe missing official terminal XY and axial conditions, respectively, and may overlap. An episode may reach its minimum XY at a different time from minimum z; comparing separate extrema cannot establish a simultaneous successful pose. Preserve paired step/state records and official success mask.

C requires directly measured relevant axis orientation; any numerical diagnostic threshold must be declared secondary/descriptive, not a new official success test or post-hoc cause. D cannot be inferred from stagnation, commanded wrench or joint reaction alone. E means official ever-success with terminal failure: it cannot occur among original/new ever-failure cases by definition, but should be checked over all episodes. Report event counts and longest/final success runs separately; ever=terminal=true does not imply continuous success in between. F is appropriate when trajectory facts do not distinguish causal mechanisms.

For comparisons, use separate tables and denominators for original G2 and new diagnostics. Explicitly count original failed IDs that are now failed/successful; describe only **same-ID outcome recurrence**, unless stronger trajectory/state matching evidence exists. Every original/new per-episode difference should be visible, including successful IDs that newly fail. Do not change original G2 evidence or primary rate. No new training was authorized.
