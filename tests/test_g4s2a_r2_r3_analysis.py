"""Negative controls for offline solver-stage review; no simulator required."""
import copy
import pytest
from scripts.analyze_g4s2a_r2_r3 import review_solver, review_accounting, contact_row_count


def solver():
    # Equality row 0; a 3D elliptic contact owns rows 1..3. A second
    # excluded contact owns none. J.T @ f = [2, 3].
    return {'nefc': 4, 'ncon': 2,
            'efc_type': [0, 7, 7, 7], 'efc_id': [2, 0, 0, 0],
            'efc_force': [2., 3., 0., 0.], 'efc_vel': [0.]*4,
            'efc_aref': [0.]*4, 'efc_pos': [0.]*4, 'efc_state': [0]*4,
            'efc_J_dense': [[1., 0.], [0., 1.], [0., 0.], [0., 0.]],
            'qfrc_constraint': [2., 3.],
            'contacts': [{'index': 0, 'dim': 3, 'efc_address': 1},
                         {'index': 1, 'dim': 3, 'efc_address': -1}]}


def test_elliptic_rows_and_force_partition():
    out = review_solver(solver(), 1)
    assert out['contact_scalar_rows'] == 3
    assert out['equality_scalar_rows'] == 1
    assert out['contacts'][0]['address'] == 1
    assert out['contacts'][0]['constraint_force'] == [3., 0., 0.]
    assert out['generalized_force_decomposition']['components'] == {'equality': [2., 0.], 'contact': [0., 3.], 'other': [0., 0.]}


@pytest.mark.parametrize('dim,cone,expected', [(1,0,1),(1,1,1),(3,1,3),(4,1,4),(6,1,6),(3,0,4),(4,0,6),(6,0,10)])
def test_constraint_dimensions(dim, cone, expected):
    assert contact_row_count(dim,cone) == expected


@pytest.mark.parametrize('field,change', [
    ('efc_id', [2,1,0,0]), ('efc_type', [0,6,7,7]),
    ('qfrc_constraint', [2.,4.]), ('efc_aref', [0.,0.])])
def test_corruption_rejected(field, change):
    data=solver(); data[field]=change
    with pytest.raises(ValueError): review_solver(data,1)


def test_excluded_contact_cannot_own_rows():
    data=solver(); data['contacts'][0]['efc_address']=-1
    with pytest.raises(ValueError): review_solver(data,1)


def test_overlapping_or_orphan_rows_rejected():
    data=solver(); data['contacts'][1]['efc_address']=1
    with pytest.raises(ValueError): review_solver(data,1)


def test_unknown_request_not_synthesized():
    events=[{'event':'integration_request','step':1}, {'event':'physics_result','step':1}, {'event':'integration_request','step':2}]
    assert review_accounting(events,1)=={'requests':2,'results':1,'unresolved_request_step':2,'complete':False}
    with pytest.raises(ValueError): review_accounting(events,2)


def test_result_without_request_fails():
    with pytest.raises(ValueError): review_accounting([{'event':'physics_result','step':1}],1)


def test_contact_force_not_misattributed_to_equality_row():
    data=solver()
    data['contacts'][0]['contact_force_local']=[3.,0.,0.,0.,0.,0.]
    review_solver(data,1)
    data['contacts'][0]['contact_force_local'][0]=2.
    with pytest.raises(ValueError): review_solver(data,1)


def test_discontinuous_durable_sequence_fails():
    events=[{'event':'integration_request','step':1,'sequence':1}, {'event':'physics_result','step':1,'sequence':3}]
    with pytest.raises(ValueError): review_accounting(events,1)
