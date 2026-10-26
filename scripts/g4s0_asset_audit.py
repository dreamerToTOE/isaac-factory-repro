#!/usr/bin/env python3
"""Read-only, CPU-only asset audit; never creates data, steps, renders or installs.

Child modes keep MuJoCo and the already-installed USD Python ABI in separate
processes.  The normal mode writes only to a NEW evidence directory and records
the subprocess status, output, time, and model provenance. No asset is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

WORK = Path('/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work')
MENAGERIE = Path('/home/ubuntu2004/mujoco_menagerie')
ISAAC = Path('/home/ubuntu2004/isaacsim-4.5.0')
USD = ISAAC / 'extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310'
BASE_URL = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/'
CACHE = Path('/home/ubuntu2004/.cache/ov/client/http')
ORIGINALS = {
    'franka_mimic.usd': CACHE / '600913b1e36513fb59ee5bb296fa3a02dc99cac8cd4659356b04bea1868f5ba8.usd',
    'factory_peg_8mm.usd': CACHE / '7f17c50be1a11dbc939c837d105bfc82c427d8d28a6a74fd88b7156b20dc8c36.usd',
    'factory_hole_8mm.usd': CACHE / 'afd8caf22b836b055fb4d39aff70b9965c6a352a9f62afe2b57c94c626c1a088.usd',
}
TABLE_URL = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd'
TABLE_CACHE = CACHE / 'd025017ca0141657e0317517b84a432f939081bdc9bdf6bcb960feededdf3d05.usd'


def file_info(path):
    p = Path(path)
    if not p.is_file():
        return {'path': str(p), 'exists': False}
    b = p.read_bytes()
    return {'path': str(p), 'exists': True, 'bytes': len(b),
            'sha256': hashlib.sha256(b).hexdigest(), 'md5': hashlib.md5(b).hexdigest()}


def emit(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def mujoco_probe(path):
    import mujoco
    t = time.monotonic()
    m = mujoco.MjModel.from_xml_path(str(path))
    name = lambda obj, i: mujoco.mj_id2name(m, obj, i)
    emit({'status': 'VERIFIED', 'scope': 'import_and_compile_candidate_only',
          'no_MjData_no_forward_no_step_no_viewer': True,
          'python': sys.executable, 'python_version': sys.version,
          'engine_version': mujoco.__version__, 'runtime_version': mujoco.mj_versionString(),
          'module': file_info(mujoco.__file__), 'model': file_info(path),
          'engine_libraries': [file_info(p) for p in Path(mujoco.__file__).parent.glob('libmujoco.so*')],
          'nq': m.nq, 'nv': m.nv, 'nu': m.nu, 'nbody': m.nbody, 'ngeom': m.ngeom,
          'joints': [{'name': name(mujoco.mjtObj.mjOBJ_JOINT, i),
                      'range': m.jnt_range[i].tolist(), 'axis': m.jnt_axis[i].tolist(),
                      'type': int(m.jnt_type[i]), 'qpos_address': int(m.jnt_qposadr[i]),
                      'dof_address': int(m.jnt_dofadr[i])} for i in range(m.njnt)],
          'bodies': [{'name': name(mujoco.mjtObj.mjOBJ_BODY, i),
                      'mass': float(m.body_mass[i]), 'center_of_mass': m.body_ipos[i].tolist(),
                      'principal_axes_wxyz': m.body_iquat[i].tolist(),
                      'diagonal_inertia': m.body_inertia[i].tolist()} for i in range(m.nbody)],
          'dof_damping': m.dof_damping.tolist(), 'dof_armature': m.dof_armature.tolist(),
          'dof_frictionloss': m.dof_frictionloss.tolist(),
          'actuators': [{'name': name(mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                         'transmission_type': int(m.actuator_trntype[i]),
                         'transmission_id': m.actuator_trnid[i].tolist(),
                         'gain_type': int(m.actuator_gaintype[i]),
                         'gain_parameters': m.actuator_gainprm[i].tolist(),
                         'bias_type': int(m.actuator_biastype[i]),
                         'bias_parameters': m.actuator_biasprm[i].tolist(),
                         'control_range': m.actuator_ctrlrange[i].tolist(),
                         'force_range': m.actuator_forcerange[i].tolist()} for i in range(m.nu)],
          'compiled_timestep': m.opt.timestep, 'compiled_gravity': m.opt.gravity.tolist(),
          'runtime_s': time.monotonic()-t,
          'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'does_not_verify': ['frozen Factory equivalence', 'closed-loop performance', 'contact behavior']})


def usd_probe():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    def val(v):
        if isinstance(v, float) and not math.isfinite(v):
            return {'authored_nonfinite_scalar': str(v)}
        if v is None or isinstance(v, (str, bool, int, float)):
            return v
        if isinstance(v, dict):
            return {str(k): val(x) for k, x in v.items()}
        if hasattr(v, 'GetReal') and hasattr(v, 'GetImaginary'):
            return [float(v.GetReal()), *map(float, v.GetImaginary())]
        if isinstance(v, Sdf.AssetPath):
            return {'asset_path': v.path, 'resolved_path': v.resolvedPath}
        try:
            if len(v) <= 64:
                return [val(x) for x in v]
            try:
                b = bytes(memoryview(v))
            except TypeError:
                b = str(v).encode()
            return {'array_length': len(v), 'sha256': hashlib.sha256(b).hexdigest(),
                    'hash_encoding': 'buffer_bytes_or_text_fallback'}
        except TypeError:
            return str(v)

    def bound(prim, stage):
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            return None
        xf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        world = [xf.Transform(Gf.Vec3d(*x)) for x in pts]
        return [[min(float(x[k]) for x in world) for k in range(3)],
                [max(float(x[k]) for x in world) for k in range(3)]]

    t = time.monotonic()
    result = {'status': 'VERIFIED', 'scope': 'read_local_self_contained_USD_only',
              'no_simulation_app_no_physics_no_gpu': True, 'usd_version': Usd.GetVersion(),
              'python': sys.executable, 'assets': []}
    for name, path in ORIGINALS.items():
        layer = Sdf.Layer.FindOrOpen(str(path))
        refs = list(layer.GetExternalReferences())
        row = {'name': name, 'file': file_info(path), 'default_prim': layer.defaultPrim,
               'external_composition_references': refs, 'sublayers': list(layer.subLayerPaths),
               'custom_layer_data': val(layer.customLayerData), 'joints': [], 'masses': [], 'geometry': []}
        if refs or layer.subLayerPaths:
            row['stage_status'] = 'BLOCKED_EXTERNAL_DEPENDENCIES_NOT_FETCHED'
            result['assets'].append(row)
            continue
        s = Usd.Stage.Open(layer, load=Usd.Stage.LoadNone)
        row['layer_metadata'] = {k: val(s.GetMetadata(k)) for k in
                                 ['metersPerUnit', 'upAxis', 'documentation', 'comment']}
        for prim in s.TraverseAll():
            attrs = {}
            for a in prim.GetAttributes():
                n = a.GetName()
                if a.HasAuthoredValue() and (n.startswith(('physics:', 'drive:', 'physx', 'xformOp'))):
                    if 'CookedData' not in n:
                        attrs[n] = val(a.Get())
            if 'Joint' in prim.GetTypeName():
                row['joints'].append({'path': str(prim.GetPath()), 'type': prim.GetTypeName(),
                                      'attributes': attrs,
                                      'relations': {str(r.GetName()): list(map(str, r.GetTargets())) for r in prim.GetRelationships()}})
            if prim.HasAPI(UsdPhysics.MassAPI):
                row['masses'].append({'path': str(prim.GetPath()), 'attributes': attrs})
            if prim.GetTypeName() == 'Mesh':
                mesh = UsdGeom.Mesh(prim)
                row['geometry'].append({'path': str(prim.GetPath()), 'attributes': attrs,
                                        'apis': list(prim.GetAppliedSchemas()),
                                        'vertex_count': len(mesh.GetPointsAttr().Get() or []),
                                        'face_count': len(mesh.GetFaceVertexCountsAttr().Get() or []),
                                        'authored_extent': val(mesh.GetExtentAttr().Get()),
                                        'raw_stage_world_bounds': bound(prim, s)})
        # References do not perform an implicit stage-unit or up-axis conversion.
        host = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(host, 1.0)
        UsdGeom.SetStageUpAxis(host, UsdGeom.Tokens.z)
        parent = UsdGeom.Xform.Define(host, '/FactoryAsset').GetPrim()
        parent.GetReferences().AddReference(str(path), '/'+layer.defaultPrim)
        row['referenced_into_Z_up_metre_host_bounds'] = {
            str(p.GetPath()): bound(p, host) for p in host.TraverseAll() if p.GetTypeName() == 'Mesh'}
        result['assets'].append(row)
    # Inspect the small cached table layer without composing its dependencies.
    table = Sdf.Layer.FindOrOpen(str(TABLE_CACHE))
    result['table_layer_only'] = {'file': file_info(TABLE_CACHE), 'url': TABLE_URL,
                                 'default_prim': table.defaultPrim,
                                 'sublayers': list(table.subLayerPaths),
                                 'external_composition_references': list(table.GetExternalReferences()),
                                 'dependency_status': 'UNKNOWN_NOT_COMPOSED_OR_FETCHED'}
    result['runtime_s'] = time.monotonic()-t
    result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    emit(result)


def candidate_inventory(model_dir):
    root = ET.parse(model_dir / ('panda.xml' if model_dir.name == 'franka_emika_panda' else 'fr3.xml')).getroot()
    references = sorted({a.get('file') for a in root.iter() if a.get('file')})
    return {'origin': str(model_dir), 'model_files': [file_info(p) for p in sorted(model_dir.glob('*.xml'))],
            'license': file_info(model_dir / 'LICENSE'), 'readme': file_info(model_dir / 'README.md'),
            'referenced_geometry': [file_info(model_dir / 'assets' / name) for name in references],
            'all_direct_model_references_exist': all((model_dir / 'assets' / name).is_file() for name in references)}


def run_probe(command, env=None):
    t = time.monotonic()
    r = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
    row = {'command': command, 'returncode': r.returncode, 'runtime_s': time.monotonic()-t,
           'stdout': r.stdout, 'stderr': r.stderr}
    if r.returncode == 0:
        row['result'] = json.loads(r.stdout)
    return row


def head(url):
    t = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=10) as r:
            return {'url': url, 'method': 'HEAD', 'status': r.status,
                    'headers': {k: r.headers.get(k) for k in ['Content-Length', 'Content-Type', 'ETag', 'Last-Modified']},
                    'body_downloaded': False, 'runtime_s': time.monotonic()-t}
    except (urllib.error.URLError, TimeoutError) as e:
        return {'url': url, 'method': 'HEAD', 'error': str(e), 'body_downloaded': False,
                'runtime_s': time.monotonic()-t}


def derived_comparison(out):
    """Compare decoded metadata; this does not assert dynamics equivalence."""
    usd = out['original_usd_probe'].get('result')
    panda = out['candidate_parse_probes'][0].get('result')
    if not usd or not panda:
        return {'status': 'BLOCKED_PROBE_FAILED'}
    robot = usd['assets'][0]
    src_joints = {j['path'].rsplit('/', 1)[-1]: j for j in robot['joints']}
    dst_joints = {j['name']: j for j in panda['joints']}
    joints = []
    for i in range(1, 8):
        s = src_joints[f'panda_joint{i}']['attributes']
        d = dst_joints[f'joint{i}']
        deg = [s['physics:lowerLimit'], s['physics:upperLimit']]
        rad = list(map(math.radians, deg))
        joints.append({'source': f'panda_joint{i}', 'candidate': f'joint{i}',
                       'source_axis': s['physics:axis'], 'candidate_local_axis': d['axis'],
                       'source_authored_limit_degree': deg, 'source_converted_limit_radian': rad,
                       'candidate_limit_radian': d['range'],
                       'maximum_limit_difference_radian': max(abs(a-b) for a, b in zip(rad, d['range'])),
                       'status': 'VERIFIED_METADATA_ONLY_NOT_FK_EQUIVALENCE'})
    src_mass = {j['path'].rsplit('/', 1)[-1]: j['attributes'].get('physics:mass') for j in robot['masses']}
    dst_mass = {j['name']: j['mass'] for j in panda['bodies']}
    masses = [{'source': f'panda_link{i}', 'candidate': f'link{i}',
               'source_mass_kg': src_mass[f'panda_link{i}'], 'candidate_mass_kg': dst_mass[f'link{i}'],
               'difference_kg': src_mass[f'panda_link{i}']-dst_mass[f'link{i}']} for i in range(8)]
    units = []
    for a in usd['assets']:
        prefix = '/'+a['default_prim']
        raw = {'/FactoryAsset'+g['path'][len(prefix):]: g['raw_stage_world_bounds'] for g in a['geometry']}
        refs = a['referenced_into_Z_up_metre_host_bounds']
        units.append({'asset': a['name'], 'layer_metadata': a['layer_metadata'],
                      'referencing_preserves_raw_geometry_bounds': raw == refs,
                      'status': 'VERIFIED_CPU_USD_REFERENCE_COMPOSITION_ONLY'})
    return {'status': 'VERIFIED_METADATA_COMPARISON_ONLY', 'joint_mapping_proposal': joints,
            'arm_link_mass_comparison': masses, 'unit_reference_controls': units,
            'source_extra_bodies': ['force_sensor', 'panda_fingertip_centered'],
            'source_finger_mass_kg': [src_mass['panda_leftfinger'], src_mass['panda_rightfinger']],
            'candidate_finger_mass_kg': [dst_mass['left_finger'], dst_mass['right_finger']],
            'dynamics_equivalence': 'UNKNOWN', 'collision_equivalence': 'UNKNOWN'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe-mujoco', type=Path)
    ap.add_argument('--probe-usd', action='store_true')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--network-head', action='store_true')
    args = ap.parse_args()
    if args.probe_mujoco:
        mujoco_probe(args.probe_mujoco)
        return
    if args.probe_usd:
        usd_probe()
        return
    if not args.output:
        ap.error('--output is required for collection')
    if args.output.exists():
        ap.error('refusing to overwrite an existing audit')
    me = str(Path(__file__).resolve())
    mjpython = '/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python'
    usd_env = os.environ.copy()
    usd_env['PYTHONPATH'] = str(USD)
    usd_env['LD_LIBRARY_PATH'] = str(USD / 'bin')
    upstream = WORK / 'upstream/IsaacLab'
    git = lambda path: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    out = {'schema_version': 1, 'task': 'G4-S0', 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'collector_source': file_info(me),
           'scope': 'read-only source/metadata, engine import and CPU candidate compilation; no physics execution',
           'repository_starting_commit': git(Path(__file__).resolve().parents[1]),
           'isaaclab_commit': git(upstream), 'menagerie_commit': git(MENAGERIE),
           'menagerie_worktree_clean': subprocess.check_output(['git', '-C', str(MENAGERIE), 'status', '--porcelain'], text=True) == '',
           'original_cached_files': {n: file_info(p) for n, p in ORIGINALS.items()},
           'candidate_inventories': [candidate_inventory(MENAGERIE / n) for n in ['franka_emika_panda', 'franka_fr3']],
           'candidate_parse_probes': [run_probe([mjpython, me, '--probe-mujoco', str(MENAGERIE / n / f)])
                                      for n, f in [('franka_emika_panda', 'panda.xml'), ('franka_fr3', 'fr3.xml')]],
           'original_usd_probe': run_probe([str(WORK / 'envs/factory-g01/bin/python'), me, '--probe-usd'], env=usd_env),
           'license_source_files': [file_info(p) for p in [
               upstream / 'LICENSE', upstream / 'docs/licenses/assets/franka-license.txt',
               upstream / 'docs/licenses/dependencies/isaacsim-license.txt',
               ISAAC / 'LICENSE.txt', ISAAC / 'PACKAGE-LICENSES/franka-LICENSE.txt',
               ISAAC / 'PACKAGE-LICENSES/isaac-sim-assets-LICENSE.txt']],
           'frozen_source_files': [file_info(upstream / 'source/isaaclab_tasks/isaaclab_tasks/direct/factory' / n)
                                   for n in ['factory_env_cfg.py', 'factory_tasks_cfg.py', 'factory_env.py']],
           'remote_head_probes': [head(BASE_URL+n) for n in [*ORIGINALS, 'LICENSE.txt', 'LICENSE', 'license.txt', 'franka-LICENSE.txt']]+[head(TABLE_URL)]
                                 if args.network_head else [],
           'primary_source_urls': {
               'mujoco_python': 'https://mujoco.readthedocs.io/en/3.5.0/python.html',
               'mujoco_license': 'https://github.com/google-deepmind/mujoco/blob/3.5.0/LICENSE',
               'panda_readme': 'https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/README.md',
               'panda_license': 'https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/LICENSE',
               'fr3_readme': 'https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_fr3/README.md',
               'isaaclab_license': 'https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/LICENSE',
               'isaaclab_franka_license': 'https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/docs/licenses/assets/franka-license.txt',
               'isaaclab_isaacsim_license': 'https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/docs/licenses/dependencies/isaacsim-license.txt',
               'current_isaac_additional_license_not_applied_retroactively': 'https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/'},
           'decision': 'CONDITIONAL', 'claim_status': {
               'existing_mujoco_engine_import_and_candidate_parse': 'VERIFIED',
               'original_cached_factory_USD_metadata_and_hashes': 'VERIFIED',
               'Menagerie_Panda_model_Apache_2_0_license_file': 'VERIFIED',
               'Menagerie_Panda_equals_Factory_franka_mimic': 'UNKNOWN',
               'Factory_assets_conversion_and_public_redistribution_permission': 'UNKNOWN',
               'faithful_collision_and_inertia_conversion': 'PROPOSED',
               'cross_engine_policy_performance': 'UNKNOWN'},
           'restrictions': {'packages_installed': False, 'substantial_assets_downloaded': False,
                            'original_assets_redistributed': False, 'physics_steps_executed': 0,
                            'policy_rollouts_executed': 0}}
    out['derived_metadata_comparison'] = derived_comparison(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')
    emit({'output': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
          'parse_returncodes': [r['returncode'] for r in out['candidate_parse_probes']],
          'usd_returncode': out['original_usd_probe']['returncode']})
    if any(r['returncode'] != 0 for r in out['candidate_parse_probes']) or out['original_usd_probe']['returncode']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
