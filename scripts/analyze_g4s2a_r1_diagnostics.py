#!/usr/bin/env python3
"""Read-only analysis of bounded GUI diagnostic records; no simulator or actor.

Plots visualize saved numerical measurements only. Static armature clarification
uses previously saved matrices and algebra, without new model compilation.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'evidence/g4_s/s2a_r1/diagnostics/gui_matrix_v1'
STATIC=ROOT/'evidence/g4_s/s2a_r1/dynamics/static_audit_20261026_v2/matched_dynamics.json'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    with Path(path).open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def rotation(q):
    w,x,y,z=np.asarray(q,dtype=float)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def relative(state):
    return rotation(state['tcp_quaternion_wxyz']).T@(np.asarray(state['peg_root_position'])-state['tcp_position_world'])
def pad_forces(state):
    force={'left':0.,'right':0.}
    for contact in state['contacts']:
        if 'peg_collision' not in (contact['geom1'],contact['geom2']):continue
        for side in force:
            if any(g in {side+'_finger_geom_'+str(i) for i in range(3,8)} for g in (contact['geom1'],contact['geom2'])):
                force[side]+=max(0.,contact['normal_force_n'])
    return force

def static_clarification(output,identities):
    static=read(STATIC);comparisons=[]
    actual=np.asarray(identities['D07']['model']['armature'])-np.asarray(identities['D01']['model']['armature'])
    expected=np.zeros_like(actual);expected[:7]=.01
    if not np.array_equal(actual,expected):raise ValueError('ACTUAL_D07_NOT_ARM7_ONLY')
    for saved in static['matched_configurations']:
        base=np.asarray(saved['variants']['frozen_s1b']['M9_mixed_rotational_translational_units'])
        armature9=np.asarray(saved['variants']['robot_joint_armature_plus_0p01']['M9_mixed_rotational_translational_units'])
        armature7=base+np.diag([.01]*7+[0.,0.])
        # Removing only the finger diagonal contribution leaves the arm7 block.
        if not np.allclose(armature7[:7,:7],armature9[:7,:7],atol=1e-14,rtol=0):raise ValueError('ARM_PRINCIPAL_BLOCK_MISMATCH')
        comparisons.append({'state_id':saved['state_id'],
          'M7_armature7_vs_armature9_max_abs_difference':float(np.max(np.abs(armature7[:7,:7]-armature9[:7,:7]))),
          'M9_armature7':armature7.tolist(),'M9_static_armature9':armature9.tolist(),
          'finger_diagonal_mass_increment_static9_kg':.01,'finger_diagonal_mass_increment_actual_D07_kg':0.,
          'effective_joint7_inertia_fingers_locked_both_cases_kg_m2':float(1/np.linalg.inv(armature7[:7,:7])[6,6]),
          'effective_joint7_inertia_full_robot_free_actual_D07_kg_m2':float(1/np.linalg.inv(armature7)[6,6]),
          'effective_joint7_inertia_full_robot_free_static9_kg_m2':float(1/np.linalg.inv(armature9)[6,6]),
          'effective_finger_inertias_full_robot_free_actual_D07_kg':(1/np.diag(np.linalg.inv(armature7)))[7:9].tolist(),
          'effective_finger_inertias_full_robot_free_static9_kg':(1/np.diag(np.linalg.inv(armature9)))[7:9].tolist(),
          'M7_condition_both_cases':float(np.linalg.cond(armature7[:7,:7])),
          'M7_local_damping_indicator_both_cases':saved['variants']['robot_joint_armature_plus_0p01']['local_velocity_feedback_indicator']['maximum_dt_eigenvalue']})
    out={'task':'G4-S2A-R1','scope':'STATIC_SCOPE_CLARIFICATION; NO NEW PHYSICS',
      'preserved_static_evidence':{'path':str(STATIC),'sha256':sha(STATIC),'modified':False},
      'actual_D07_identity_sha256':sha(DEFAULT/'D07/identity.json'),
      'static_v1_v2_armature_scope':'7 revolute arm coordinates plus2 prismatic finger coordinates. Rotational armature units kg m²; prismatic coordinate mass increment units kg. This was algebraic sensitivity only, not an integrated experiment.',
      'actual_registered_D07_scope':'7 revolute arm coordinates ONLY, +0.01 kg m² per coordinate; finger armature unchanged. Corrected before registration and integration.',
      'actual_armature_delta_vector':actual.tolist(),'comparisons':comparisons,
      'interpretation':['The arm7 principal M block and finger-locked M7-based figures from static v2 also apply to actual D07 at those matched q.','The full9-coordinate matrices and unconstrained responses differ because static9 also increases both finger-coordinate masses. Do not describe the static9 experiment as the actual D07 physical variant.','Neither armature sensitivity equals source body rotational +0.01I. The body-inertia pullback is dense and depends on angular Jacobians.','Actual equality/contact-constrained dynamics are not predicted by these unconstrained static matrices.'],
      'new_model_compilations':0,'new_physics_steps':0,'new_actor_calls':0}
    write(output/'static_armature_scope_clarification.json',out)
    return out


def analyze(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=False)
    summary=read(source/'summary.json');registry=read(source/'preregistered_matrix.json')
    if summary['actual_physics_steps']!=160 or len(summary['receipts'])!=10 or registry['maximum_registered_steps']!=160:raise ValueError('EXPECTED_BOUNDED_SCOPE_NOT_FOUND')
    if any(r['exact_child_exit_code']!=0 for r in summary['receipts']):raise ValueError('NONZERO_CHILD_EXIT')
    records={};identities={};aggregates=[];hashes={};max_slip_error=0.;max_force_error=0.;all_warnings=0
    for receipt in summary['receipts']:
        vid=receipt['variant'];directory=source/vid;identity=read(directory/'identity.json');identities[vid]=identity
        result=read(directory/'result.json');lines=[json.loads(x) for x in (directory/'physics.jsonl').read_text().splitlines()]
        requests=[x for x in lines if x['event']=='integration_request'];rows=[x for x in lines if x['event']=='physics_result']
        if len(rows)!=16 or len(requests)!=16 or result['actual_physics_steps']!=16:raise ValueError('DIAGNOSTIC_STEP_COUNT')
        if [x['step'] for x in requests]!=list(range(1,17)) or [x['step'] for x in rows]!=list(range(1,17)):raise ValueError('DIAGNOSTIC_STEP_SEQUENCE')
        if any(lines[i]['event']!=('integration_request' if i%2==0 else 'physics_result') for i in range(32)):raise ValueError('REQUEST_RESULT_ORDER')
        initial=identity['initial_snapshot'];ref=relative(initial);slips=[];forces=[];qvel=[];warnings=[]
        if vid!='D01':
            for key in ('qpos','qvel'):
                if not np.array_equal(initial[key],identities['D01']['initial_snapshot'][key]):raise ValueError('INITIAL_STATE_CHANGED:'+vid)
        for step,row in enumerate(rows,1):
            state=row['state'];slip=float(np.linalg.norm(relative(state)-ref));force=pad_forces(state)
            if not all(np.isfinite(np.asarray(state[key])).all() for key in ('qpos','qvel')):raise ValueError('NONFINITE_STATE')
            max_slip_error=max(max_slip_error,abs(slip-state['s2a']['peg_hand_slip_m']))
            max_force_error=max(max_force_error,max(abs(force[k]-row['pad_force_n'][k]) for k in force))
            if abs(row['episode_time_s']-step/120)>1e-11:raise ValueError('PHYSICS_CLOCK_GAP')
            slips.append(slip);forces.append(force);qvel.append(max(abs(np.asarray(state['qvel'][:7]))));warnings.extend(row['new_warning_counts'])
        recomputed={'max_slip_m':max(slips),'max_arm_qvel_rad_s':max(qvel),'first_bilateral_loss_step':next((i+1 for i,f in enumerate(forces) if min(f.values())<=1e-6),None),
          'max_all_penetration_m':max(row['state']['s2a']['all_contact_penetration_m'] for row in rows)}
        for key in ('max_slip_m','max_arm_qvel_rad_s','max_all_penetration_m'):
            if not math.isclose(recomputed[key],result[key],rel_tol=1e-11,abs_tol=1e-12):raise ValueError('SUMMARY_METRIC_MISMATCH:'+vid+':'+key)
        if recomputed['first_bilateral_loss_step']!=result['first_bilateral_loss_step']:raise ValueError('CONTACT_LOSS_MISMATCH')
        all_warnings+=sum(warnings)
        aggregate={'variant':vid,'name':result['variant']['name'],'scope':'DIAGNOSTIC_ONLY','model_configuration_sha256':result['model_configuration_sha256'],
          **recomputed,'max_arm_limit_rad':result['max_arm_limit_rad'],'max_finger_limit_m':result['max_finger_limit_m'],'recorded_new_warning_count':sum(warnings),
          'exact_child_exit_code':receipt['exact_child_exit_code'],'physical_steps':16,'time_horizon_s':16/120,'policy_calls':0}
        records[vid]=rows;aggregates.append(aggregate)
        for name in ('identity.json','result.json','physics.jsonl'):
            path=directory/name;hashes[str(path.relative_to(ROOT))]=sha(path)
    if max_slip_error>1e-9 or max_force_error!=0:raise ValueError('INDEPENDENT_MEASUREMENT_RECONSTRUCTION_FAILED')
    clarification=static_clarification(output,identities)
    a={r['variant']:r for r in aggregates};ratio=a['D01']['max_slip_m']/a['D06']['max_slip_m'];velocity_ratio=a['D01']['max_arm_qvel_rad_s']/a['D06']['max_arm_qvel_rad_s']
    hypotheses=[
      {'id':'H1','hypothesis':'机器人惯量缺失或不匹配','assessment':'SUPPORTED_BUT_NOT_PROVEN','confidence':'HIGH_WITHIN_16_STEP_DIAGNOSTIC_SCOPE',
       'verified_facts':['Factory源码明确对所有返回刚体转动惯量加0.01I；冻结S1B未实施这一数值增量。','映射11刚体+0.01I的D06在16步维持双侧正接触；最大滑移88.56µm，D01为14.241mm。'],
       'controlled_experiments':['D01 vs D06','D06 vs D07'],
       'quantitative':{'D01_to_D06_max_slip_ratio':ratio,'D01_to_D06_max_arm_velocity_ratio':velocity_ratio},
       'contradicting_or_limiting_evidence':['缺失source sensor和centered TCP刚体，原始PhysX cooked惯量及实际M均UNKNOWN。','惯量同时改变物理响应和质量相关nullspace运算；不是整机忠实重建或最终根因证明。','D06全接触峰值穿透13.32µm包含指垫接触；不能直接套用peg-hole接触的10µm判据。16步抓持保持不等于插入成功。']},
      {'id':'H2','hypothesis':'力矩饱和及过大加速度','assessment':'SUPPORTED_BUT_NOT_PROVEN','confidence':'HIGH_FOR_AMPLIFICATION_NOT_SOLE_CAUSE',
       'verified_facts':['历史第2物理步已明显速度反向，早于第3步joint7首次达到12Nm限制。','D08放宽至100Nm后16步滑移约34982.55m；状态数值有限且无MuJoCo警告，仍明显物理失真。'],
       'controlled_experiments':['D01 vs D08'],
       'contradicting_or_limiting_evidence':['仅由饱和触发最早不稳定这一子假设被时间顺序反驳。','不能把configured PhysX effort limits当作实测PhysX施力。','D08无警告只说明引擎未发出该类警告，不表示物理有效。']},
      {'id':'H3','hypothesis':'任务空间PD与离散反馈/低惯量耦合','assessment':'SUPPORTED_BUT_NOT_PROVEN','confidence':'HIGH_WITHIN_LOCAL_WINDOW',
       'verified_facts':['D04仅任务力矩仍在第4步失去双侧接触；D05仅nullspace力矩16步未失去。','D03固定初始TCP参考仍在第8步失去接触，故策略持续更新或每步重建目标不是唯一必要条件。','匹配初态理想阻尼指标max(dt λ(M^-1 JᵀKdJ))：D01=13.812，D06=1.958，D07=6.277。'],
       'controlled_experiments':['D01 vs D03','D04 vs D05','D01 vs D06 vs D07'],
       'contradicting_or_limiting_evidence':['局部显式Euler阻尼指标忽略FD延迟、饱和、接触及实际积分器，不能作为全系统稳定证明。','未改变dt/KP/KD进行额外因果分离；不自动调参。']},
      {'id':'H4','hypothesis':'Jacobian/nullspace实现或动态输入失配','assessment':'INCONCLUSIVE','confidence':'CHECKED_ALGEBRA_SUPPORTED; PRIMARY_IMPLEMENTATION_BUG_NOT_SUPPORTED',
       'verified_facts':['两匹配q的source-authored FK Jacobian与MuJoCo均值finger-origin Jacobian最大差约1.35e-8。','动力学一致nullspace残差在已检查姿态小于4e-14；历史前8步joint7 nullspace力矩接近零。','D05保留nullspace驱动仍16步保持双侧抓持；D04去掉nullspace仍第4步失败。'],
       'controlled_experiments':['D04 vs D05','matched static J/M/nullspace audit'],
       'contradicting_or_limiting_evidence':['这些证据不支持nullspace实现错误是最初wrist振荡的主因。','原始PhysX Jacobian/M未测量，source-authored FK不是引擎运行时量；质量模型差异会改变投影。']},
      {'id':'H5','hypothesis':'手指执行器或接触求解响应','assessment':'SUPPORTED_BUT_NOT_PROVEN','confidence':'MODERATE; LOAD_DEPENDENT_GRASP_REMAINS_UNRESOLVED',
       'verified_facts':['D02零臂驱动维持抓持，最大滑移2.79µm；D10小幅接触软化也维持，2.05µm。','D09零臂驱动但改为affine finger PD，第3步失去双侧接触，峰值穿透33.61mm。'],
       'controlled_experiments':['D02 vs D09 vs D10','D01 vs D02'],
       'contradicting_or_limiting_evidence':['冻结step-implicit PD并非在该初态无负载时必然失稳。','D09失败不能推出其他affine调参均失败；也不能把它作为修复。','未独立改变强臂加速度下的抓持力/摩擦，不能排除弱抓持与臂加速度共同作用。']},
      {'id':'H6','hypothesis':'初始抓持/reset分布失配','assessment':'INCONCLUSIVE','confidence':'LOW_FOR_CROSS_DISTRIBUTION_CAUSALITY',
       'verified_facts':['十变体采用完全相同的归档初始qpos/qvel；D02/D06/D10在各自16步窗口可保持抓持。','S2A初始物理准入通过，使用本地NumPy噪声流及固定hole/support姿态，非配对PhysX reset。'],
       'controlled_experiments':['same-state D01/D02/D06 comparisons'],
       'contradicting_or_limiting_evidence':['初态本身不足以导致所有变体同样失败，但不排除原Factory分布差异。','只有一个种子/一个姿态，没有分布统计或复现率结论；warmstart与求解器内部状态未完整归档。']},
      {'id':'H7','hypothesis':'观测、四元数或有限差分速度因素','assessment':'INCONCLUSIVE','confidence':'NO_INTERFACE_BUG_EVIDENCE; FD_FEEDBACK_COUPLING_SUPPORTED',
       'verified_facts':['本轮仅重用归档动作，无任何新策略推理；S2A独立审计已核对raw/RMS/action/EMA与源码。','历史前8步符号/速度变化对应真实关节运动；任务PD导数项随速度反向显著放大。'],
       'controlled_experiments':['source parity audit','D03/D04 local feedback diagnostics'],
       'contradicting_or_limiting_evidence':['未做独立FD替换/时间步变更或四元数输入扰动，所以不能把FD实现判为错误或排除其离散耦合。','目标引擎姿态与原PhysX原始四元数符号运行时对应关系仍未直接测量。']}
    ]
    evidence={'task':'G4-S2A-R1','scope':'READ_ONLY_ANALYSIS_OF_DIAGNOSTIC_ONLY_16_STEP_WINDOWS','new_physics_steps':0,'new_actor_calls':0,'model_compilations':0,
      'source_directory':str(source),'source_sha256':{'diagnostic_preregistration':sha(source/'preregistered_matrix.json'),'diagnostic_summary':sha(source/'summary.json'),'analysis_script':sha(__file__)},
      'checked_counts':{'variants':10,'integration_requests':160,'physics_results':160,'exact_successful_child_exits':10,'actor_calls':0,'warning_count':all_warnings},
      'independent_checks':{'all_initial_qpos_qvel_exactly_equal':True,'all_request_result_sequences_and_clocks_valid':True,'slip_world_pose_reconstruction_max_abs_error_m':max_slip_error,'pad_force_reconstruction_max_abs_error_N':max_force_error,'all_summary_extrema_match_reconstructed':True},
      'variant_results':aggregates,'source_file_digests':hashes,'hypotheses':hypotheses,
      'root_cause_status':'ROOT_CAUSE_PENDING','solution_status':'SOLUTION_PENDING','validation_status':'FULL_POLICY_TASK_VALIDATION_PENDING',
      'bounded_verified_conclusion':'With identical archived initial qpos/qvel and archived action inputs, the mapped-body rotational inertia diagnostic D06 retained bilateral positive finger contact over16steps and greatly reduced slip/arm velocity compared with D01. This does not establish faithful Factory dynamics or successful zero-shot PPO insertion.',
      'limitations':['One16step window per variant, not repeated-trial performance statistics.','Zero new policy inference; action sequence is open-loop archived input, not counterfactual closed-loop PPO behavior.','No original PhysX runtime mass matrix or applied actuator output measured.','D06 has13.32µm peak all-contact penetration including finger pads. The10µm insertion-contact criterion applies to peg-hole assembly contacts, not all gripper contacts; no insertion success is established.','D08 is physically invalid despite finite state and zero warnings; exit0 is process completion only.']}
    write(output/'comparison.json',evidence)
    with (output/'variant_comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True,constrained_layout=True)
    for vid,color in [('D01','#c94343'),('D06','#197743'),('D07','#2b67ad')]:
        data=records[vid];steps=np.arange(1,17)
        axes[0].plot(steps,[r['state']['qvel'][6] for r in data],'-o',label=vid,color=color,markersize=3)
        axes[1].semilogy(steps,[r['state']['s2a']['peg_hand_slip_m']*1000 for r in data],'-o',label=vid,color=color,markersize=3)
        axes[2].plot(steps,[min(pad_forces(r['state']).values()) for r in data],'-o',label=vid,color=color,markersize=3)
    axes[0].axhline(0,color='grey',linewidth=.6);axes[0].set_ylabel('Joint7 velocity [rad/s]');axes[0].legend(ncol=3)
    axes[1].set_ylabel('Peg-hand slip [mm, log]');axes[1].axhline(1,color='grey',linestyle=':',label='1mm reference only')
    axes[2].set_yscale('symlog',linthresh=.001);axes[2].set_ylabel('Min left/right pad force [N]');axes[2].set_xlabel('New diagnostic physics step [dt=1/120s]')
    for ax in axes:ax.grid(True,alpha=.25);ax.set_xticks(range(1,17))
    fig.suptitle('DIAGNOSTIC_ONLY: archived actions, identical initial q/qdot; 16-step windows\nD01 frozen / D06 body+0.01I / D07 arm7 armature+0.01; no policy inference')
    fig.savefig(output/'body_inertia_vs_armature.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    ax.bar([x['variant'] for x in aggregates],[x['max_slip_m'] for x in aggregates],color=['#197743' if x['variant'] in ('D02','D05','D06','D10') else '#c94343' for x in aggregates])
    ax.set_yscale('log');ax.set_ylabel('Maximum peg-hand slip [m, log]');ax.set_title('DIAGNOSTIC_ONLY: all10 registered16-step windows\nD08 finite-but-unphysical divergence retained; no task-success interpretation')
    ax.grid(True,axis='y',alpha=.25);fig.savefig(output/'all_variant_slip.png',dpi=160);plt.close(fig)
    write(output/'manifest.json',{'files':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},'new_physics_steps':0,'new_actor_calls':0})
    print(json.dumps({'variants':10,'new_physics_steps':0,'D01_to_D06_slip_ratio':ratio,'D01_to_D06_velocity_ratio':velocity_ratio,'reconstruction_error_m':max_slip_error,'output':str(output)}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.source,a.output)
if __name__=='__main__':main()
