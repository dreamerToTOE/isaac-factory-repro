#!/usr/bin/env python3
"""Static FK projection of archived contact midpoints onto frozen pad boxes.

No integration, viewer or actor. A contact midpoint is not a surface witness;
local box margins are descriptive only and do not prove an edge collision.
"""
import gzip,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evidence/g4_s/s2a_r2/matrix_v2'
OUT=ROOT/'evidence/g4_s/s2a_r2_r1/forensics_v1/geometry_projection.json'
XML=BASE/'P1_diagnostic.xml'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=mujoco.MjModel.from_xml_path(str(XML)); d=mujoco.MjData(m)
 result={'method':'Frozen P1 static FK via mj_forward only; no mj_step or viewer. Recorded contact midpoint projected into box coordinates at same archived poststep qpos.','limitations':'Midpoints do not identify actual collision surface feature or prove physical contact at edge/corner. Negative margin means midpoint outside box span, not an independently established manifold defect. Saved solver impulses unavailable.','model_xml_sha256':sha(XML),'mujoco_version':mujoco.__version__,'rows':{}}
 for initial,variant,lo,hi in [('I3','B',150,177),('I3','D',190,216),('I2','B',150,177),('I2','D',190,216),('I1','B',150,177),('I1','D',190,216)]:
  rid=initial+'_'+variant;rows=[]
  with gzip.open(BASE/'trajectories'/rid/'physics.jsonl.gz','rt') as f:
   for line in f:
    x=json.loads(line)
    if x['event']!='physics_result' or not lo<=x['step']<=hi:continue
    s=x['state'];d.qpos[:]=s['qpos'];d.qvel[:]=s['qvel'];mujoco.mj_forward(m,d)
    contacts=[]
    for c in s['contacts']:
     geom=next((g for g in (c['geom1'],c['geom2']) if '_finger_geom_' in g),None)
     if not geom:continue
     gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,geom)
     if gid<0 or m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_BOX: continue
     R=np.array(d.geom_xmat[gid]).reshape(3,3); local=R.T@(np.array(c['position_world_m'])-d.geom_xpos[gid]); half=m.geom_size[gid]
     contacts.append({'geom':geom,'local_midpoint_m':local.tolist(),'half_size_m':half.tolist(),
       'in_plane_x_z_signed_margin_m':(half[[0,2]]-np.abs(local[[0,2]])).tolist(),
       'distance_to_y_face_m':float(abs(abs(local[1])-half[1])),
       'normal_force_n':c['normal_force_n'],'distance_m':c['distance_m']})
    rows.append({'step':x['step'],'contacts':contacts})
  result['rows'][rid]=rows
 with OUT.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 for rid in ('I3_B','I3_D','I2_B','I2_D'):
  rows=result['rows'][rid]
  print(rid)
  for x in rows[-4:]:
   print(x['step'],[(c['geom'],[round(v*1e6,1) for v in c['in_plane_x_z_signed_margin_m']]) for c in x['contacts']])
if __name__=='__main__':main()
