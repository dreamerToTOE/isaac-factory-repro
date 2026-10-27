#!/usr/bin/env python3
"""Static MuJoCo 3.5.0 API/constraint audit for G4-S2A-R2-R2.

This script compiles the already-frozen P_FULL XML and only inspects model
metadata and Python API contracts. It does not create a policy, call mj_step,
or call mj_forward.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys
import mujoco

ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--xml',type=pathlib.Path,default=ROOT/'evidence/g4_s/s2a_r2_r1/matrix_v2/P_FULL_diagnostic.xml'); p.add_argument('--output',type=pathlib.Path,required=True); a=p.parse_args()
    xml=a.xml.resolve(); model=mujoco.MjModel.from_xml_path(str(xml))
    selected=['mj_contactForce','mj_fullM','mj_forward','mj_step']
    api={name:(getattr(mujoco,name).__doc__ or '').strip() for name in selected}
    contacts=['dist','efc_address','frame','geom1','geom2','pos','dim','friction','solimp','solref']
    data_fields=['contact','efc_address','efc_type','efc_id','efc_force','efc_pos','efc_vel','qfrc_constraint','qacc','qacc_warmstart','actuator_force','qfrc_actuator','nefc','warning']
    eq_type={name:int(getattr(mujoco.mjtEq,name)) for name in dir(mujoco.mjtEq) if name.startswith('mjEQ_')}
    cnstr={name:int(getattr(mujoco.mjtConstraint,name)) for name in dir(mujoco.mjtConstraint) if name.startswith('mjCNSTR_')}
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(i)) for i in range(model.njnt)]
    eqs=[]
    for i in range(model.neq):
        eqs.append({'index':i,'type':int(model.eq_type[i]),'obj1id':int(model.eq_obj1id[i]),'obj2id':int(model.eq_obj2id[i]),'data':model.eq_data[i].tolist()})
    out={'schema_version':1,'task':'G4-S2A-R2-R2','scope':'STATIC_API_AND_COMPILED_MODEL_ONLY','mujoco_version':mujoco.__version__,'mj_version_string':mujoco.mj_versionString(),'xml_path':str(xml.relative_to(ROOT)) if xml.is_relative_to(ROOT) else str(xml),'xml_sha256':hashlib.sha256(xml.read_bytes()).hexdigest(),'physics_calls':{'mj_step':0,'mj_forward':0,'mj_contactForce':0,'mj_fullM':0},'api_docstrings':api,'MjContact_fields_present':{x:hasattr(mujoco.MjContact,x) for x in contacts},'MjData_fields_present':{x:hasattr(mujoco.MjData,x) for x in data_fields},'constraint_constants':cnstr,'equality_constants':eq_type,'compiled_model':{'nq':int(model.nq),'nv':int(model.nv),'nu':int(model.nu),'njnt':int(model.njnt),'neq':int(model.neq),'joint_names':names,'equality_constraints':eqs,'timestep_s':float(model.opt.timestep),'integrator':int(model.opt.integrator),'solver':int(model.opt.solver),'iterations':int(model.opt.iterations),'tolerance':float(model.opt.tolerance)},'semantics':['contact.efc_address indexes the constraint array for a contact; it is not a contact-array index.','mj_contactForce returns a six-dimensional force/torque in the contact frame for an existing contact id.','efc_type/efc_id/efc_force/efc_pos/efc_vel are constraint-space arrays ordered by active constraint rows, not by raw contact rows.','mj_forward computes forward dynamics without integrating time; this audit did not call it.','post-step mj_forward values in historical traces are recomputed derived quantities and are not historical integration impulses.']}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'mujoco_version':out['mujoco_version'],'neq':model.neq,'physics_calls':out['physics_calls']})); return 0
if __name__=='__main__': raise SystemExit(main())
