"""Small CPU audit calculations, not a target simulator or policy runner.

These functions make interface hypotheses testable. They neither issue commands
nor implement a MuJoCo environment/controller. Engine/sign/asset parity is S1 work.
Geometry/action calculations use float64 as mathematical audit fixtures, not a
bitwise PhysX implementation. Observation assembly and success use source FP32.
"""
from __future__ import annotations
import math
import numpy as np

ARM_NAMES=tuple(f'panda_joint{i}' for i in range(1,8))
PHYSICS_DT=1/120
DECIMATION=8


def vector(value,n,name):
    out=np.asarray(value)
    if out.shape!=(n,) or out.dtype.kind!='f' or not np.isfinite(out).all():
        raise ValueError(f'{name}: expected finite floating [{n}]')
    return out.astype(np.float64,copy=True)


def quaternion(value):
    out=vector(value,4,'wxyz quaternion')
    if abs(np.linalg.norm(out)-1)>1e-6:
        raise ValueError('quaternion must be unit length; no silent normalization')
    return out


def wxyz_to_xyzw(value):
    return quaternion(value)[[1,2,3,0]]


def xyzw_to_wxyz(value):
    out=vector(value,4,'xyzw quaternion')[[3,0,1,2]]
    return quaternion(out)


def quat_mul(a,b):
    # Internal products may include the original sign(0) zero quaternion edge.
    w,x,y,z=a;v,i,j,k=b
    return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])


def world_point(position,orientation,local_offset):
    p=vector(position,3,'position');q=quaternion(orientation)
    v=vector(local_offset,3,'local offset')
    return p+quat_mul(quat_mul(q,np.r_[0.,v]),q*np.array([1.,-1.,-1.,-1.]))[1:]


def fd_velocity(position,previous_position,orientation,previous_orientation,*,dt=PHYSICS_DT,reset=False):
    if not isinstance(reset,bool) or not math.isfinite(dt) or dt!=PHYSICS_DT:
        raise ValueError('frozen physics dt/reset contract mismatch')
    p=vector(position,3,'position');old=vector(previous_position,3,'previous position')
    q=quaternion(orientation);prev=quaternion(previous_orientation)
    if reset:return np.zeros(3),np.zeros(3)
    delta=quat_mul(q,prev*np.array([1.,-1.,-1.,-1.]))
    delta*=np.sign(delta[0])  # preserve pinned sign(0) behavior
    sine=np.linalg.norm(delta[1:]);half=math.atan2(sine,delta[0])
    angle=2*half
    divisor=(.5-angle*angle/48) if abs(angle)<=1e-6 else math.sin(half)/angle
    return (p-old)/dt,delta[1:]/divisor/dt


def observation(position,orientation,fixed_reset_tip,episode_noise,linear_fd,angular_fd,ema):
    p=vector(position,3,'position');q=quaternion(orientation)
    frame=vector(fixed_reset_tip,3,'fixed reset tip');noise=vector(episode_noise,3,'noise')
    relative=p.astype(np.float32)-(frame.astype(np.float32)+noise.astype(np.float32))
    result=np.concatenate([relative,q,vector(linear_fd,3,'linear FD'),
                           vector(angular_fd,3,'angular FD'),vector(ema,6,'EMA')]).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('FP32 observation overflow')
    return result


def translation_action(raw_mu,previous_cache,current_position,fixed_action_frame):
    mu=vector(raw_mu,6,'raw mu');old=vector(previous_cache,6,'cache')
    p=vector(current_position,3,'position');frame=vector(fixed_action_frame,3,'action frame')
    clipped=np.clip(mu,-1.,1.)
    cache=.2*clipped+.8*old
    position=frame+np.clip(p+.02*cache[:3]-frame,-.05,.05)
    return {'clipped':clipped,'ema':cache,'position_target':position,'world_rotvec_increment':.097*cache[3:]}


def reset_translation_cache(position,fixed_action_frame):
    return (vector(position,3,'position')-vector(fixed_action_frame,3,'action frame'))/.05


def joint_indices(target_joint_names,source_to_target):
    """Only an explicit indexing audit; never proves kinematic equivalence."""
    names=tuple(target_joint_names)
    if len(names)!=len(set(names)) or set(source_to_target)!=set(ARM_NAMES):
        raise ValueError('duplicate target names or incomplete frozen arm mapping')
    requested=tuple(source_to_target[name] for name in ARM_NAMES)
    if len(set(requested))!=7 or any(name not in names for name in requested):
        raise ValueError('mapping must identify seven distinct named target joints')
    return tuple(names.index(name) for name in requested)


def official_success(held_root,fixed_root):
    """PegInsert roots have zero base offsets. Strict one-sided z predicate."""
    held=vector(held_root,3,'held root').astype(np.float32)
    fixed=vector(fixed_root,3,'fixed root').astype(np.float32)
    delta=held-fixed
    if not np.isfinite(delta).all():raise ValueError('FP32 success input overflow')
    return bool(np.linalg.norm(delta[:2])<np.float32(.0025) and delta[2]<np.float32(.025*.04))


def timing(physics_dt=PHYSICS_DT,decimation=DECIMATION):
    if isinstance(decimation,bool) or type(decimation)!=int or decimation!=8 or physics_dt!=PHYSICS_DT:
        raise ValueError('timing differs from frozen 120Hz / decimation8')
    return {'physics_dt':physics_dt,'policy_dt':physics_dt*decimation,'policy_hz':1/(physics_dt*decimation),
            'timeout_policy_steps':149,'timeout_simulation_seconds':149*physics_dt*decimation}
