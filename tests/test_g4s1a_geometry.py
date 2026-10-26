"""Independent rational geometry oracles and recorded actual-engine diagnostics."""
from fractions import Fraction as F
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from scripts import g4s1a_geometry_audit as g

ROOT=Path(__file__).resolve().parents[1]
TETRA_POINTS=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
TETRA_FACES=[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]


def test_segment_extrema_include_interior_not_just_vertices():
    assert g.segment_radius_squared((1,-1),(1,1))==(F(1),F(2))
    assert g.segment_radius_squared((-1,0),(1,0))==(F(0),F(1))
    assert g.segment_radius_squared((2,3),(2,3))==(F(13),F(13))


@pytest.mark.parametrize('value,places',[(F(2),15),(F(1,9),18),(F(0),5),(F(1,10**40),30),(F(25),12)])
def test_directed_sqrt_brackets_are_checked_by_exact_integer_square(value,places):
    lo,hi=g.sqrt_decimal_bracket(value,places)
    assert lo*lo<=value<=hi*hi
    assert hi-lo<=F(1,10**places)


def test_tetrahedron_topology_independent_known_euler_and_missing_face():
    stat=g.validate_triangle_mesh(TETRA_POINTS,TETRA_FACES)
    assert stat['closed_oriented_triangle_surface']
    assert stat['euler_characteristic_used']==2
    assert stat['boundary_edges']==0
    bad=g.validate_triangle_mesh(TETRA_POINTS,TETRA_FACES[:-1])
    assert not bad['closed_oriented_triangle_surface']
    assert bad['boundary_edges']==3


def test_duplicate_faces_and_flipped_orientation_are_not_certified_closed():
    assert not g.validate_triangle_mesh(TETRA_POINTS,TETRA_FACES+[TETRA_FACES[0]])['closed_oriented_triangle_surface']
    bad=TETRA_FACES.copy();bad[0]=tuple(reversed(bad[0]))
    assert not g.validate_triangle_mesh(TETRA_POINTS,bad)['closed_oriented_triangle_surface']


@pytest.mark.parametrize('points,faces,error',[
    ([],[], 'EMPTY_MESH'),
    ([(math.nan,0,0)],[(0,0,0)],'INVALID_VERTEX'),
    ([(math.inf,0,0)],[(0,0,0)],'INVALID_VERTEX'),
    ([(0,0)],[(0,0,0)],'INVALID_VERTEX'),
    (TETRA_POINTS,[(0,1,4)],'INVALID_FACE'),
    (TETRA_POINTS,[(0,1,-1)],'INVALID_FACE'),
    (TETRA_POINTS,[(0,1)],'INVALID_FACE'),
])
def test_invalid_assets_rejected_before_compilation(points,faces,error):
    with pytest.raises(ValueError,match=error):g.validate_triangle_mesh(points,faces)


def test_section_cuts_tetrahedron_with_exact_nonvertex_radial_minimum():
    # z=1/2 triangle has vertices (0,0),(.5,0),(0,.5).
    section=g.section_measurement(TETRA_POINTS,TETRA_FACES,F(1,2))
    assert F(section['minimum_radius_squared_exact'])==0
    assert F(section['maximum_radius_squared_exact'])==F(1,4)
    assert section['selected_segments']==3
    with pytest.raises(ValueError,match='THROUGH_VERTEX'):
        g.section_measurement(TETRA_POINTS,TETRA_FACES,F(0))


def test_axis_parallel_triangle_is_reported_ambiguous_not_silently_missed():
    with pytest.raises(ValueError,match='AXIS_PARALLEL_SURFACE_AMBIGUITY'):
        g.axial_surface_hits(TETRA_POINTS,[(0,1,3)])


def test_full_slab_projection_checks_triangle_interior_not_selected_z_samples():
    # Horizontal triangle at z=.01 contains the axis in its XY interior.
    pts=[(-1,-1,.01),(1,-1,.01),(0,1,.01)]
    wall=g.inner_wall_slab(pts,[(0,1,2)])
    assert F(wall['minimum_radius_squared_exact'])==0
    # All edge points have positive radius; a vertex-only check would miss this.
    assert all(x*x+y*y>0 for x,y,z in pts)


def test_analytic_bore_error_does_not_erase_fifty_seven_micrometer_gap_silently():
    coarse,fine=g.sector_error_bounds(16),g.sector_error_bounds(64)
    assert coarse['bore_error_upper_bound_m']>57e-6
    assert coarse['minimum_nominal_radial_clearance_lower_bound_m']<0
    assert fine['bore_error_upper_bound_m']<5e-6
    assert fine['minimum_nominal_radial_clearance_lower_bound_m']>52e-6
    bounds=[F(g.sector_error_bounds(n)['bore_error_upper_bound_exact'])for n in (16,32,64,128)]
    assert all(a==4*b for a,b in zip(bounds,bounds[1:]))


def test_custom_float_radii_preserve_exact_rational_bound_fields():
    bounds=g.sector_error_bounds(64,.00405,.02)
    expected=F(.00405)*F(22,7*64)**2/2
    assert F(bounds['bore_error_upper_bound_exact'])==expected
    assert '/'in bounds['bore_error_upper_bound_exact']


@pytest.mark.parametrize('n',[-1,0,1,3,4.5])
def test_bad_sector_count_rejected(n):
    with pytest.raises(ValueError,match='INVALID_ANNULUS_PARAMETERS'):g.sector_error_bounds(n)


def test_generated_prisms_remain_closed_with_open_axis_and_no_weld():
    root=ET.fromstring(g.annulus_xml(8))
    assert 'not_factory'in root.get('model')
    assert root.find('equality')is None
    assert len(root.findall('./asset/mesh'))==8
    for mesh in root.findall('./asset/mesh'):
        vs=list(map(float,mesh.get('vertex').split()))
        fs=list(map(int,mesh.get('face').split()))
        points=[tuple(vs[i:i+3])for i in range(0,len(vs),3)]
        faces=[tuple(fs[i:i+3])for i in range(0,len(fs),3)]
        assert g.validate_triangle_mesh(points,faces)['closed_oriented_triangle_surface']
        assert all(math.hypot(x,y)>.004 for x,y,z in points)


@pytest.fixture(scope='module')
def evidence():
    return json.loads((ROOT/'evidence/g4_s/s1a/geometry_audit.json').read_text())


def test_raw_source_hashes_and_full_slab_discrepancy_are_preserved(evidence):
    original=evidence['original_geometry_reference']
    assert original['returncode']==0
    assert all(a['original_unchanged']for a in original['result']['assets'])
    wall=original['result']['assets'][1]['meshes'][0]['complete_straight_inner_wall']
    assert wall['status']=='EXACT_RATIONAL_ALL_BOUNDARY_SLAB_BOUND'
    assert wall['triangle_count']==588
    assert F(wall['minimum_radius_squared_exact'])>F(4498,1000000)**2
    reconcile=evidence['source_nominal_dimension_reconciliation']
    assert reconcile['coaxial_raw_geometric_radial_gap_lower_bound_m']>505e-6
    assert reconcile['coaxial_raw_gap_lower_bound_divided_by_nominal57um']>8.8
    assert evidence['error_budget']['original_57um_fidelity_certified']is False
    assert evidence['error_budget']['source_cooked_PhysX_SDF_to_raw_mesh_m']=='UNKNOWN'


def test_actual_mujoco_static_negative_control_and_invalid_assets(evidence):
    probe=evidence['independent_synthetic_compile']
    assert probe['returncode']==0
    actual=probe['result']
    assert actual['engine_version']=='3.5.0'
    assert actual['reproducible_compiled_geometry']
    assert all(a['rejected']for a in actual['invalid_assets'])
    opened,wall=actual['static_synthetic_distance_checks']
    assert opened['contact_count']==0 and opened['minimum_geom_distance_m']>.0039
    assert wall['contact_count']>0 and wall['minimum_geom_distance_m']<0
    bad=actual['single_convex_hull_negative_control']
    assert bad['bore_integrity_rejected']and bad['axis_sphere_distance_m']<-.01
    assert all(row['physics_time_s']==0 for row in [opened,wall,bad])
    assert actual['physics_steps']==actual['policy_inferences']==0
    assert actual['compiled_vertex_quantization']['maximum_bidir_vertex_distance_m_approx']<2e-9
