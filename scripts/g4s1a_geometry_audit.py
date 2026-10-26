#!/usr/bin/env python3
"""G4-S1A CPU geometry gate; no original mesh export, physics or policy execution.

The original USD probe emits only scalar measurements and topology statistics.
The annulus is independent mathematics, NOT a Factory conversion or replacement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1] / 'work'
USD_LIB = Path('/home/ubuntu2004/isaacsim-4.5.0/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310')
USD_PYTHON = WORK / 'envs/factory-g01/bin/python'
MJ_PYTHON = Path('/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python')
MENAGERIE = Path('/home/ubuntu2004/mujoco_menagerie/franka_emika_panda')
NOMINAL_GAP = Fraction(57, 1_000_000)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational(value):
    return value if isinstance(value, Fraction) else Fraction(value)


def sqrt_decimal_bracket(value, places=15):
    """Directed exact rational sqrt bracket; approximate display is separate."""
    value=rational(value)
    if value < 0 or not isinstance(places,int) or places < 0:
        raise ValueError('INVALID_SQRT_BRACKET')
    scale=10**places
    lower=Fraction(math.isqrt((value.numerator*scale*scale)//value.denominator),scale)
    upper=lower if lower*lower==value else lower+Fraction(1,scale)
    return lower,upper


def segment_radius_squared(a, b):
    """Exact squared radial extrema for a 2D rational line segment."""
    a, b = tuple(map(rational, a)), tuple(map(rational, b))
    d = tuple(y-x for x, y in zip(a, b))
    dot = lambda x, y: sum((v*w for v, w in zip(x, y)), Fraction())
    den = dot(d, d)
    t = max(Fraction(), min(Fraction(1), -dot(a, d)/den)) if den else Fraction()
    c = tuple(x+t*v for x, v in zip(a, d))
    return dot(c, c), max(dot(a, a), dot(b, b))


def validate_triangle_mesh(points, faces):
    """Reject invalid numeric/topological input before any target compilation."""
    if not points or not faces:
        raise ValueError('EMPTY_MESH')
    if any(len(p) != 3 or any(not math.isfinite(float(x)) for x in p) for p in points):
        raise ValueError('INVALID_VERTEX')
    if any(len(f) != 3 or any(not isinstance(i, int) or i < 0 or i >= len(points) for i in f) for f in faces):
        raise ValueError('INVALID_FACE')
    welded = {p: i for i, p in enumerate(dict.fromkeys(tuple(p) for p in points))}
    edges, directed = Counter(), Counter()
    zero_area = 0
    used = set()
    for face in faces:
        ids = tuple(welded[tuple(points[i])] for i in face)
        a, b, c = [tuple(map(rational, points[i])) for i in face]
        ab, ac = [b[k]-a[k] for k in range(3)], [c[k]-a[k] for k in range(3)]
        cross = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]]
        if not any(cross):
            zero_area += 1
        used.update(ids)
        for i, j in zip(ids, ids[1:]+ids[:1]):
            edges[tuple(sorted((i, j)))] += 1
            directed[i, j] += 1
    return {'vertices': len(points), 'exact_coordinate_welded_vertices': len(welded), 'faces': len(faces),
            'used_vertices': len(used), 'undirected_edges': len(edges),
            'edge_incidence_histogram': dict(sorted(Counter(edges.values()).items())),
            'boundary_edges': sum(n == 1 for n in edges.values()),
            'nonmanifold_edges': sum(n > 2 for n in edges.values()), 'zero_area_faces': zero_area,
            'orientation_inconsistent_edges': sum(directed[i,j] != 1 or directed[j,i] != 1 for (i,j), n in edges.items() if n == 2),
            'euler_characteristic_used': len(used)-len(edges)+len(faces),
            'closed_oriented_triangle_surface': bool(edges) and all(n == 2 for n in edges.values()) and zero_area == 0 and all(directed[i,j] == directed[j,i] == 1 for (i,j), n in edges.items())}


def section_measurement(points, faces, z, radial_cutoff=None):
    """All triangle/plane intersections, exact rationals; never finite angle samples.

    Planes through a mesh vertex are refused, avoiding coplanar ambiguity.  A
    radial cutoff only labels the inner-wall subset; it is never a collision filter.
    """
    z = rational(z)
    ps = [tuple(map(rational, p)) for p in points]
    if any(p[2] == z for p in ps):
        raise ValueError('SECTION_PLANE_THROUGH_VERTEX')
    segments = set()
    for f in faces:
        vs = [ps[i] for i in f]
        hits = []
        for a, b in zip(vs, vs[1:]+vs[:1]):
            if (a[2]-z)*(b[2]-z) < 0:
                u = (z-a[2])/(b[2]-a[2])
                hits.append(tuple(a[k]+u*(b[k]-a[k]) for k in range(2)))
        if len(hits) == 2:
            segments.add(tuple(sorted(hits)))
    selected = []
    cutoff2 = rational(radial_cutoff)**2 if radial_cutoff else None
    for seg in segments:
        lo, hi = segment_radius_squared(*seg)
        if cutoff2 is None or hi < cutoff2:
            selected.append((lo, hi))
    if not selected:
        return {'z_m': float(z), 'all_section_segments': len(segments), 'selected_segments': 0, 'status': 'NO_SELECTED_BOUNDARY'}
    lo, hi = min(x[0] for x in selected), max(x[1] for x in selected)
    return {'z_m': float(z), 'z_exact': str(z), 'all_section_segments': len(segments),
            'selected_segments': len(selected), 'selection_radial_cutoff_m': radial_cutoff,
            'minimum_radius_m_approx': math.sqrt(float(lo)), 'maximum_radius_m_approx': math.sqrt(float(hi)),
            'minimum_radius_squared_exact': str(lo), 'maximum_radius_squared_exact': str(hi),
            'minimum_radius_above_nominal_hole_radius': lo > Fraction(405,100000)**2,
            'status': 'EXACT_RATIONAL_POLYGON_SECTION_EXTREMA_SQRT_DISPLAY_ONLY'}


def axial_surface_hits(points, faces):
    """Exact axis/triangle plane intersections, aggregated as z coordinates only."""
    hits = set()
    for f in faces:
        a,b,c = [tuple(map(rational, points[i])) for i in f]
        area = lambda p,q,r: (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
        den = area(a,b,c)
        if not den:
            if min(segment_radius_squared(x[:2],y[:2])[0]for x,y in ((a,b),(b,c),(c,a)))==0:
                raise ValueError('AXIS_PARALLEL_SURFACE_AMBIGUITY')
            continue
        o = (Fraction(),Fraction(),Fraction())
        weights = (area(o,b,c)/den,area(a,o,c)/den,area(a,b,o)/den)
        if min(weights) >= 0:
            hits.add(sum((weights[i]*p[2] for i,p in enumerate((a,b,c))),Fraction()))
    return [{'z_m': float(z), 'z_exact': str(z)} for z in sorted(hits)]


def inner_wall_slab(points, faces):
    """Exact lower radial bound of ALL mesh boundary in the open bore slab.

    Clip every triangle that intersects the open z slab to its closed slab.
    Project the resulting convex polygon onto XY, testing origin containment
    and every edge. This covers triangle interiors and all intermediate z;
    the bore floor at z=0 and chamfer at z>=.024 are deliberately separate.
    """
    low, high = Fraction(), Fraction(.024000000208616257)
    selected, stats = [], []
    for face in faces:
        vs = [tuple(map(rational,points[i]))for i in face]
        if max(p[2]for p in vs)<=low or min(p[2]for p in vs)>=high:
            continue
        for cut,above in ((low,True),(high,False)):
            clipped=[]
            for a,b in zip(vs,vs[1:]+vs[:1]):
                ina=a[2]>=cut if above else a[2]<=cut
                inb=b[2]>=cut if above else b[2]<=cut
                if ina:
                    clipped.append(a)
                if ina!=inb:
                    u=(cut-a[2])/(b[2]-a[2])
                    clipped.append(tuple(a[k]+u*(b[k]-a[k])for k in range(3)))
            vs=clipped
        if not vs:
            continue
        selected.append(face)
        edge_stats=[segment_radius_squared(x[:2],y[:2])for x,y in zip(vs,vs[1:]+vs[:1])]
        crosses=[a[0]*b[1]-a[1]*b[0]for a,b in zip(vs,vs[1:]+vs[:1])]
        contains_origin=sum(crosses)!=0 and (min(crosses)>=0 or max(crosses)<=0)
        stats.append((Fraction()if contains_origin else min(x[0]for x in edge_stats),max(x[1]for x in edge_stats)))
    if not selected:
        return {'status':'INCONCLUSIVE_WALL_NOT_FOUND'}
    lo,hi=min(x[0]for x in stats),max(x[1]for x in stats)
    return {'status':'EXACT_RATIONAL_ALL_BOUNDARY_SLAB_BOUND',
            'triangle_count':len(selected),'z_interval_m':[float(low),float(high)],
            'minimum_radius_squared_exact':str(lo),'maximum_radius_squared_exact':str(hi),
            'minimum_radius_m_approx':math.sqrt(float(lo)),'maximum_radius_m_approx':math.sqrt(float(hi)),
            'scope':'All raw mesh boundary triangles intersecting open z slab(0,.024); exact rational clipping/projection covers entire triangle interiors, not finitely selected z planes.',
            'selection':'ALL triangles intersecting open z slab; floor/chamfer at slab endpoints excluded explicitly; no radial cutoff.'}


def sector_error_bounds(n, inner=Fraction(405,100000), outer=Fraction(2,100)):
    """Analytic conservative annulus cover; pi < 22/7 makes rational upper bounds.

    The inner regular polygon is inscribed (no bore enlargement). The outer
    polygon is circumscribed (no material shrinkage). Sector pieces are convex
    trapezoidal prisms. These are bounds for ideal analytic mathematical solids;
    float serialization/compiled hull error is explicitly NOT certified here.
    """
    inner,outer=rational(inner),rational(outer)
    if not isinstance(n,int) or n < 4 or not 0 < inner < outer:
        raise ValueError('INVALID_ANNULUS_PARAMETERS')
    x = Fraction(22,7*n)
    a = x*x/2
    bore = inner*a                 # 1-cos(x) <= x^2/2
    exterior = outer*a/(1-a)       # sec(x)-1 <= a/(1-a)
    return {'sectors': n, 'inner_boundary_sagitta_m_approx': float(inner)*(1-math.cos(math.pi/n)),
            'bore_error_upper_bound_m': float(bore), 'bore_error_upper_bound_exact': str(bore),
            'outer_boundary_error_upper_bound_m': float(exterior), 'outer_boundary_error_upper_bound_exact': str(exterior),
            'bore_error_bound_fraction_of_nominal_gap': float(bore/NOMINAL_GAP),
            'minimum_nominal_radial_clearance_lower_bound_m': float(NOMINAL_GAP-bore),
            'ideal_solid_hausdorff_upper_bound_m': float(max(bore,exterior)),
            'bound_scope': 'ideal_independent_analytic_annulus_not_original_mesh_or_compiled_hulls'}


def annulus_xml(n=64, inner=.00405, outer=.02, height=.025):
    """Independent synthetic prism union. No original Factory coordinates read."""
    sector_error_bounds(n,Fraction(inner),Fraction(outer))
    if not math.isfinite(height) or height <= 0:
        raise ValueError('INVALID_HEIGHT')
    meshes, geoms = [], []
    # The four vertices are ordered anticlockwise; cap faces and sides close it.
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    for j in range(n):
        t0,t1 = 2*math.pi*j/n,2*math.pi*(j+1)/n
        ro = outer/math.cos(math.pi/n)
        xy = [(inner*math.cos(t0),inner*math.sin(t0)),(ro*math.cos(t0),ro*math.sin(t0)),
              (ro*math.cos(t1),ro*math.sin(t1)),(inner*math.cos(t1),inner*math.sin(t1))]
        points = [(x,y,z) for z in (0,height) for x,y in xy]
        stat = validate_triangle_mesh(points,faces)
        if not stat['closed_oriented_triangle_surface']:
            raise ValueError('INVALID_GENERATED_PRISM')
        fmt = lambda rows:' '.join(format(x,'.17g') if isinstance(x,float) else str(x) for row in rows for x in row)
        meshes.append(f'<mesh name="sector{j}" vertex="{fmt(points)}" face="{fmt(faces)}"/>')
        geoms.append(f'<geom name="sector{j}" type="mesh" mesh="sector{j}"/>')
    # One moving sphere allows optional STATIC collision distance checks. No step.
    return '<mujoco model="independent_annulus_not_factory"><compiler angle="radian" usethread="false"/>'+\
           '<option gravity="0 0 0"/><asset>'+''.join(meshes)+'</asset><worldbody><body name="independent_ring">'+\
           ''.join(geoms)+'</body><body name="test_sphere" pos="0 0 .0125"><freejoint/>'+\
           '<geom name="probe" type="sphere" size=".0001" mass=".001"/></body></worldbody></mujoco>'


def usd_probe(s0_path):
    from pxr import Gf, Sdf, Usd, UsdGeom
    s0 = json.loads(Path(s0_path).read_text())
    out = {'scope':'READ_ONLY_ORIGINAL_USD_SCALAR_GEOMETRY_METADATA_NO_CONVERSION_NO_VERTEX_EXPORT','usd_version':Usd.GetVersion(),'assets':[]}
    for name in ('factory_peg_8mm.usd','factory_hole_8mm.usd'):
        ref = s0['original_cached_files'][name]
        path = Path(ref['path'])
        if sha256(path) != ref['sha256']:
            raise ValueError('ORIGINAL_HASH_MISMATCH:'+name)
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer.GetExternalReferences() or layer.subLayerPaths:
            raise ValueError('UNRESOLVED_ORIGINAL_DEPENDENCY:'+name)
        stage = Usd.Stage.Open(layer,load=Usd.Stage.LoadNone)
        asset = {'name':name,'sha256_before':sha256(path),'meshes':[]}
        for prim in stage.TraverseAll():
            if prim.GetTypeName()!='Mesh':
                continue
            mesh = UsdGeom.Mesh(prim)
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            if transform != Gf.Matrix4d(1):
                raise ValueError('NONIDENTITY_ORIGINAL_GEOMETRY_TRANSFORM_REQUIRES_REVIEW')
            points = [tuple(map(float,transform.Transform(Gf.Vec3d(*p)))) for p in mesh.GetPointsAttr().Get()]
            counts,indices = list(mesh.GetFaceVertexCountsAttr().Get()),list(mesh.GetFaceVertexIndicesAttr().Get())
            if any(c != 3 for c in counts):
                raise ValueError('NONTRIANGLE_ORIGINAL_REQUIRES_REVIEW')
            faces = [tuple(map(int,indices[i:i+3]))for i in range(0,len(indices),3)]
            zs = sorted(set(p[2]for p in points))
            sections = [Fraction(1,1000),Fraction(125,10000),Fraction(239,10000),Fraction(245,10000)] if 'hole' in name else [Fraction(1,1000),Fraction(25,1000),Fraction(49,1000)]
            cooked=[]
            for attr in prim.GetAttributes():
                if 'CookedData' in attr.GetName() and attr.HasAuthoredValue():
                    value=attr.Get()
                    cooked.append({'name':attr.GetName(),'type':str(attr.GetTypeName()),'length':len(value)if hasattr(value,'__len__')else None,'decoded':False})
            asset['meshes'].append({'path':str(prim.GetPath()),'topology':validate_triangle_mesh(points,faces),
                'local_to_world_transform_verified_identity':True,
                'maximum_raw_mesh_radius_squared_exact':str(max(Fraction(p[0])**2+Fraction(p[1])**2 for p in points)),
                'world_bounds_m':[[min(p[k]for p in points)for k in range(3)],[max(p[k]for p in points)for k in range(3)]],
                'vertex_z_levels_m':zs,'sections':[section_measurement(points,faces,z,.008 if 'hole'in name else None)for z in sections],
                'axis_surface_intersections':axial_surface_hits(points,faces),
                'complete_straight_inner_wall':inner_wall_slab(points,faces)if 'hole'in name else None,
                'collision_approximation':prim.GetAttribute('physics:approximation').Get(),
                'cooked_buffers':cooked,'original_cooked_SDF_effective_surface_error_m':'UNKNOWN_NOT_DECODED_OR_PHYSX_EXECUTED'})
        asset['sha256_after']=sha256(path)
        asset['original_unchanged']=asset['sha256_before']==asset['sha256_after']
        out['assets'].append(asset)
    return out


def mujoco_probe():
    import mujoco
    if mujoco.__version__!='3.5.0' or mujoco.mj_versionString()!='3.5.0':
        raise ValueError('MUJOCO_VERSION_MISMATCH')
    xml=annulus_xml()
    models=[]
    for _ in range(2):
        start=time.monotonic()
        m=mujoco.MjModel.from_xml_string(xml)
        # Deterministic key compiled geometry arrays, not simulation state.
        keys=['mesh_vert','mesh_face','mesh_vertadr','mesh_vertnum','mesh_faceadr','mesh_facenum','geom_type','geom_size','geom_pos','geom_quat','geom_dataid']
        digest=hashlib.sha256()
        for key in keys:
            v=getattr(m,key)
            digest.update(key.encode());digest.update(v.dtype.str.encode());digest.update(str(v.shape).encode());digest.update(v.tobytes())
        models.append({'nbody':m.nbody,'ngeom':m.ngeom,'nmesh':m.nmesh,'nq':m.nq,'nv':m.nv,
                       'compiled_geometry_sha256':digest.hexdigest(),'compile_s':time.monotonic()-start})
    # Compare all source synthetic prism vertices to compiled transformed ones.
    # This is a measured floating calculation, not a directed-rounding bound.
    import numpy as np
    tree=ET.fromstring(xml)
    quantization=[]
    for asset in tree.findall('./asset/mesh'):
        name=asset.get('name')
        mid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_MESH,name)
        gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)
        source=np.fromstring(asset.get('vertex'),sep=' ').reshape(-1,3)
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,m.geom_quat[gid])
        compiled=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]@rotation.reshape(3,3).T+m.geom_pos[gid]
        distances=np.linalg.norm(source[:,None,:]-compiled[None,:,:],axis=2)
        quantization.append(float(max(distances.min(axis=0).max(),distances.min(axis=1).max())))
    # Independent synthetic geometry prerequisites passed above. STATIC only.
    d=mujoco.MjData(m)
    probe=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'probe')
    distance_rows=[]
    for label,x in [('axis_open_bore',0.),('inner_wall_contact',.00404)]:
        d.qpos[0]=x;mujoco.mj_forward(m,d)
        distances=[mujoco.mj_geomDistance(m,d,gid,probe,.1,None)for gid in range(64)]
        distance_rows.append({'case':label,'sphere_center_m':[x,0,.0125],'sphere_radius_m':.0001,
                              'minimum_geom_distance_m':min(distances),'contact_count':d.ncon,'physics_time_s':d.time})
    # Ordinary collision uses the convex hull even if all ring vertices survive.
    vertices=' '.join(asset.get('vertex')for asset in tree.findall('./asset/mesh'))
    hull_xml='<mujoco model="independent_bad_single_hull"><asset><mesh name="sealed" vertex="'+vertices+'"/></asset><worldbody><geom name="sealed" type="mesh" mesh="sealed"/><body pos="0 0 .0125"><freejoint/><geom name="probe" type="sphere" size=".0001" mass=".001"/></body></worldbody></mujoco>'
    hm=mujoco.MjModel.from_xml_string(hull_xml);hd=mujoco.MjData(hm);mujoco.mj_forward(hm,hd)
    hull={'scope':'INDEPENDENT_SYNTHETIC_RING_STANDARD_SINGLE_MESH_GEOM_NEGATIVE_CONTROL',
          'axis_sphere_distance_m':mujoco.mj_geomDistance(hm,hd,0,1,.1,None),'contact_count':hd.ncon,
          'bore_integrity_rejected':hd.ncon>0,'physics_time_s':hd.time}
    invalid=[]
    for name,bad in [('zero_radius','<mujoco><worldbody><geom type="sphere" size="0"/></worldbody></mujoco>'),
                     ('missing_mesh','<mujoco><worldbody><geom type="mesh" mesh="missing"/></worldbody></mujoco>'),
                     ('coplanar_mesh','<mujoco><asset><mesh name="flat" vertex="0 0 0 1 0 0 0 1 0 1 1 0"/></asset><worldbody><geom type="mesh" mesh="flat"/></worldbody></mujoco>')]:
        try:
            mujoco.MjModel.from_xml_string(bad)
            invalid.append({'name':name,'rejected':False})
        except ValueError as exc:
            invalid.append({'name':name,'rejected':True,'actual_error':str(exc)})
    return {'scope':'INDEPENDENT_SYNTHETIC_ANNULUS_COMPILATION_AND_STATIC_COLLISION_ONLY_NOT_FACTORY','engine_version':mujoco.__version__,
            'xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),'repeat_compilation':models,
            'reproducible_compiled_geometry':models[0]['compiled_geometry_sha256']==models[1]['compiled_geometry_sha256'],
            'compiled_vertex_quantization':{'maximum_bidir_vertex_distance_m_approx':max(quantization),
                'mesh_vert_dtype':str(m.mesh_vert.dtype),'scope':'Measured all64 synthetic meshes after compiler recenter/rotation; double arithmetic, not rigorous rounding bound or Factory conversion error.'},
            'static_synthetic_distance_checks':distance_rows,'single_convex_hull_negative_control':hull,
            'invalid_assets':invalid,'physics_steps':0,'policy_inferences':0,'MjData_created':True,
            'compiled_hull_error_to_ideal_annulus':'NOT_CERTIFIED_NOT_FACTORY_ERROR'}


def run_child(command,env=None):
    started=time.monotonic()
    cp=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
    out={'command':list(map(str,command)),'returncode':cp.returncode,'runtime_s':time.monotonic()-started,'stderr':cp.stderr}
    if cp.returncode==0:
        out['result']=json.loads(cp.stdout)
    else:
        out['stdout']=cp.stdout
    return out


def collect(s0_path):
    s0=json.loads(Path(s0_path).read_text())
    ctrl=json.loads((ROOT/'evidence/g4_s/s0/controller_audit.json').read_text())
    robot=s0['original_usd_probe']['result']['assets'][0]
    finger=[g for g in robot['geometry']if any(n in g['path']for n in ('leftfinger','rightfinger','panda_hand'))and 'PhysicsCollisionAPI'in g['apis']]
    panda=ET.parse(MENAGERIE/'panda.xml').getroot()
    pad=[]
    for node in panda.iter('default'):
        if (node.get('class')or'').startswith('fingertip_pad_collision'):
            pad.append({'class':node.get('class'),'geom':node.find('geom').attrib})
    env=os.environ.copy();env['PYTHONPATH']=str(USD_LIB);env['LD_LIBRARY_PATH']=str(USD_LIB/'bin')
    original=run_child([str(USD_PYTHON),str(Path(__file__).resolve()),'--probe-usd','--s0',str(s0_path)],env)
    synthetic=run_child([str(MJ_PYTHON),str(Path(__file__).resolve()),'--probe-mujoco'])
    original_dimensions={'status':'INCONCLUSIVE_ORIGINAL_PROBE_FAILED'}
    if original['returncode']==0:
        assets=original['result']['assets']
        peg=assets[0]['meshes'][0];hole=assets[1]['meshes'][0]
        wall=hole['complete_straight_inner_wall']
        if wall['status']=='EXACT_RATIONAL_ALL_BOUNDARY_SLAB_BOUND':
            rlo,_=sqrt_decimal_bracket(Fraction(wall['minimum_radius_squared_exact']))
            _,rhi=sqrt_decimal_bracket(Fraction(peg['maximum_raw_mesh_radius_squared_exact']))
            clearance=rlo-rhi
            original_dimensions={'status':'VERIFIED_RAW_SOURCE_GEOMETRY_DISAGREES_WITH_NOMINAL_CONFIGURATION',
                'straight_bore_radius_lower_bound_m':float(rlo),'straight_bore_radius_lower_bound_exact':str(rlo),
                'peg_maximum_radius_upper_bound_m':float(rhi),'peg_maximum_radius_upper_bound_exact':str(rhi),
                'coaxial_raw_geometric_radial_gap_lower_bound_m':float(clearance),'coaxial_raw_geometric_radial_gap_lower_bound_exact':str(clearance),
                'coaxial_raw_gap_lower_bound_divided_by_nominal57um':float(clearance/NOMINAL_GAP),
                'scope':'Straight raw bore slab z in (0,.024) and complete raw peg maximum radius at coaxial vertical placement; no cooked SDF/contact or arbitrary pose/physics claim.',
                'do_not_correct_by_rescaling':'Keep original unmodified. Resolve source/config/cooked mismatch before a fidelity claim; do not enlarge/shrink either part.'}
    return {'schema_version':1,'task':'G4-S1A','generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            's0_evidence_sha256':sha256(s0_path),'collector_sha256':sha256(__file__),
            'scope':'CPU_FIRST_GEOMETRY_PREREQUISITES_NO_ORIGINAL_ASSET_CONVERSION_OR_REDISTRIBUTION',
            'nominal_dimensions_m':{'peg_diameter':.007986,'hole_diameter':.0081,'radial_gap':float(NOMINAL_GAP)},
            'original_geometry_reference':original,'independent_synthetic_compile':synthetic,
            'source_nominal_dimension_reconciliation':original_dimensions,
            'candidate_comparison':[
                {'method':'single_convex_hull','decision':'REJECTED','reason':'The convex hull of a full ring contains its axis and fills the physical bore; visual mesh nonconvexity does not alter standard convex mesh collision.'},
                {'method':'bore_preserving_convex_decomposition','decision':'CONDITIONAL_NOT_CONVERTED','original_error_m':'UNKNOWN','requirements':'Permission; original inner wall, chamfer, floor and outer support surfaces; conservative or two-sided source-relative Hausdorff bound and per-section clearance; compiled-hull validation.'},
                {'method':'native_triangle_visual_mesh_as_standard_mesh_geom','decision':'REJECTED_AS_FAITHFUL_NONCONVEX_COLLIDER','reason':'Standard mesh collisions use the convex hull; triangle preservation in the asset is insufficient.'},
                {'method':'MuJoCo_SDF_plugin','decision':'CONDITIONAL_NOT_IMPLEMENTED','original_error_m':'UNKNOWN','requirements':'Licensed faithful source SDF signed-distance and gradient, independently bounded fit/grid interpolation and solver tolerance; cooked PhysX bytes are not a portable SDF API.'},
                {'method':'ideal_cylinder_plus_ideal_hole','decision':'REJECTED_AS_FACTORY_SUBSTITUTION','reason':'Nominal dimensions differ from measured source bore and omit chamfer/floor/outer details; independent diagnostics only.'}],
            'analytic_independent_annulus_diagnostics':{'identity':'ORIGINAL_MATHEMATICAL_DIAGNOSTIC_NOT_DERIVED_FROM_FACTORY_ASSET','distribution':'Generated source code only; no copied Factory vertices or meshes; repository license applies to code, third-party rights not inferred.',
                'reference':'Ideal circular annulus rin=.00405,rout=.02,height=.025; independent nominal-dimensional diagnostic, not a Factory reconstruction.',
                'bound_derivation':'Inner inscribed N-gon: rin*(1-cos(pi/N)) <= rin*(22/(7N))^2/2. Outer circumscribed N-gon: rout*(sec(pi/N)-1) <= rout*a/(1-a), a=(22/(7N))^2/2. Union covers ideal solid, retains an open bore. No claim of compiled floating-hull error or original geometry error.',
                'candidates':[sector_error_bounds(n)for n in (16,32,64,128,256,512,1024)]},
            'gripper_reconstruction':{'status':'NOT_RECONSTRUCTED_CONTACT_EQUIVALENCE_UNKNOWN','original_collision_metadata':finger,'menagerie_independent_pad_boxes':pad,
                'source_finger_mass_kg':s0['derived_metadata_comparison']['source_finger_mass_kg'],'candidate_finger_mass_kg':s0['derived_metadata_comparison']['candidate_finger_mass_kg'],
                'source_finger_control':next(x['original']for x in ctrl['physics_compatibility_matrix']if x['parameter']=='finger_actuation'),
                'physical_grasp_requirements':['Peg remains independent 6-DOF rigid body with finger/peg contacts; no hand weld.','Original contact surface dimensions, relative frames and materials must be mapped, not replaced by Menagerie five-pad boxes without comparison.','Reset closes fingers for .25 simulated seconds with original gains and randomized pose distribution; no probe performed here.','Finger mimic, implicit PD saturation and contact compliance require dedicated validation.'],
                'peg_weld_created':False,'original_geometry_converted':False},
            'support_geometry':{'status':'BLOCKED_FOR_FAITHFUL_RECONSTRUCTION','table_dependency':s0['original_usd_probe']['result']['table_layer_only'],
                'original_hole_fixed_joint':'Source fixed-world joint retained as a mapping requirement; no altered collider built.','requirements':'Resolve table USD dependency and applicable license, collision transforms, full dimensions and material before any Factory physics probe.'},
            'error_budget':{'source_raw_mesh_to_converted_mesh_m':'UNKNOWN_NO_CONVERSION','source_cooked_PhysX_SDF_to_raw_mesh_m':'UNKNOWN','source_cooked_PhysX_SDF_to_MuJoCo_collision_m':'UNKNOWN','original_57um_fidelity_certified':False},
            'restrictions':{'original_assets_copied':False,'original_derived_mesh_exported':False,'original_dimensions_modified':False,'original_geometry_preserved':True,'physics_steps':0,'policy_rollouts':0,'checkpoint_loads':0},
            'primary_sources':{'mujoco_mesh':'https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html#asset-mesh','mujoco_collision':'https://mujoco.readthedocs.io/en/3.5.0/computation/index.html#collision-detection','mujoco_sdf':'https://mujoco.readthedocs.io/en/3.5.0/programming/extension.html#sdf'},
            'diagnostic_result':'PASS_INDEPENDENT_SYNTHETIC_CAPABILITY_ONLY',
            'decision':'NO-GO_FOR_S1B_FAITHFUL_FACTORY_MODEL'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s0',type=Path,default=ROOT/'evidence/g4_s/s0/assets_audit.json')
    p.add_argument('--probe-usd',action='store_true');p.add_argument('--probe-mujoco',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    started=time.monotonic()
    if a.probe_usd:
        out=usd_probe(a.s0)
    elif a.probe_mujoco:
        out=mujoco_probe()
    else:
        if not a.output or a.output.exists():
            p.error('--output must name a new file')
        out=collect(a.s0)
    out['runtime_s']=time.monotonic()-started;out['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data=json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(data)
    else:
        print(data,end='')

if __name__=='__main__':
    main()
