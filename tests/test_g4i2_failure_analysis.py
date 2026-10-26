"""Small synthetic CPU controls; no simulator or actual failure aggregation."""
import csv
import hashlib
import json
from copy import deepcopy

import pytest

from scripts.g4i2_failure_analysis import analyze_runs, threshold_provenance


def frozen_identity(tmp_path):
    config=tmp_path/'factory_tasks_cfg.py'
    config.write_text('class Hole8mm:\n    height = 0.025\nclass Peg8mm:\n    height = 0.050\nclass PegInsert:\n    success_threshold: float = 0.04\n')
    environment=tmp_path/'factory_env.py'
    environment.write_text('def _get_curr_successes():\n    is_centered = torch.where(xy_dist < 0.0025, one, zero)\n')
    identity=tmp_path/'identity.json'
    identity.write_text(json.dumps(dict(source_inventory=[dict(label=path.name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in (config,environment)])))
    return identity


def run_fixture(tmp_path,mutate=None):
    run=tmp_path/'run';run.mkdir()
    rows=[]
    for step in (1,2,3):
        rows.append(dict(kind='action',policy_step=step,raw_mu=[[.5]*6]*2,clipped=[[.5]*6]*2,ema=[[.1]*6]*2))
        held=[[.0003,0,.0005 if step==2 else .02],[.003,0,.02]]
        rows.append(dict(kind='success_state',policy_step=step,capture=dict(success=[step==2,False],step_numbers=[step,step],geometric_snapshots=dict(held_base_pos=held,target_held_base_pos=[[0.,0.,0.]]*2))))
    if mutate: mutate(rows)
    (run/'i2_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    episodes=[dict(evaluation_seed=1001,env_id=index,episode_id=0,steps=3,ever_success=index==0,
                   terminal_success=False,first_success_step=2 if index==0 else None,
                   pre_reset_terminal_state=dict(provenance='synthetic_test_only'),
                   initial_randomized_configuration=dict(provenance='synthetic_test_only'),
                   trace=dict(first_policy_step=1,last_policy_step=3)) for index in (0,1)]
    (run/'episodes.json').write_text(json.dumps(episodes))
    return run


def test_frozen_sources_checked_and_strict_signed_thresholds(tmp_path):
    path=frozen_identity(tmp_path)
    result=threshold_provenance(path)
    assert result['z_threshold_m'] == .001
    assert result['xy_threshold_m'] == .0025
    assert result['fixed_hole_height_m'] == .025
    assert result['success_threshold_fraction'] == .04
    (tmp_path/'factory_env.py').write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'): threshold_provenance(path)


def test_failure_per_step_csv_and_independent_geometry(tmp_path):
    identity=frozen_identity(tmp_path);run=run_fixture(tmp_path)
    result=analyze_runs([run],tmp_path/'out',identity_path=identity)
    assert result['failed_episodes']==2
    first,second=result['failures']
    assert first['temporal_failure_class']=='lost_success_before_terminal'
    assert second['temporal_failure_class']=='never_success'
    assert first['terminal_xy_distance_m']==pytest.approx(.0003)
    assert first['terminal_signed_z_displacement_m']==pytest.approx(.02)
    assert first['minimum_signed_z_displacement_m']==pytest.approx(.0005)
    assert first['terminal_unmet_criteria']==['SIGNED_Z_NOT_BELOW_0_001_M']
    assert len(second['terminal_unmet_criteria'])==2
    assert first['diagnostic_predicate_mismatches']==0
    with open(first['csv_path']) as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==3
    assert rows[1]['official_success']=='True'
    assert rows[1]['raw_mu_0']=='0.5'
    assert rows[1]['clipped_5']=='0.5'
    assert rows[1]['ema_5']=='0.1'
    assert first['raw_trace']['sha256']==hashlib.sha256((run/'i2_trace.jsonl').read_bytes()).hexdigest()
    assert len(list((tmp_path/'out').iterdir()))==3


@pytest.mark.parametrize('kind',['action','success_state'])
def test_missing_failure_step_rejects_before_output(tmp_path,kind):
    identity=frozen_identity(tmp_path)
    run=run_fixture(tmp_path,mutate=lambda rows:rows.__setitem__(slice(None),[row for row in rows if not(row['kind']==kind and row['policy_step']==2)]))
    with pytest.raises(ValueError,match='incomplete failure evidence'):
        analyze_runs([run],tmp_path/'out',identity_path=identity)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind',['action','success_state'])
def test_duplicate_records_rejected(tmp_path,kind):
    identity=frozen_identity(tmp_path)
    def mutation(rows):
        index=next(i for i,row in enumerate(rows) if row['kind']==kind)
        rows.insert(index,deepcopy(rows[index]))
    run=run_fixture(tmp_path,mutate=mutation)
    with pytest.raises(ValueError,match='duplicate failure'):
        analyze_runs([run],tmp_path/'out',identity_path=identity)


def test_original_flags_remain_primary_and_diagnostic_disagreement_visible(tmp_path):
    identity=frozen_identity(tmp_path)
    def mutation(rows):
        for row in rows:
            if row['kind']=='success_state' and row['policy_step']==3:
                row['capture']['geometric_snapshots']['held_base_pos'][0]=[.0001,0,.0005]
    run=run_fixture(tmp_path,mutate=mutation)
    result=analyze_runs([run],tmp_path/'out',identity_path=identity)
    failure=result['failures'][0]
    assert failure['terminal_success'] is False
    assert failure['terminal_unmet_criteria']==[]
    assert failure['diagnostic_predicate_mismatches']==1
    assert failure['terminal_recomputed_state']['diagnostic_success'] is True


def test_signed_negative_z_is_not_replaced_with_absolute_value(tmp_path):
    identity=frozen_identity(tmp_path)
    def mutation(rows):
        for row in rows:
            if row['kind']=='success_state' and row['policy_step']==2:
                row['capture']['geometric_snapshots']['held_base_pos'][0]=[.0001,0,-.02]
    run=run_fixture(tmp_path,mutate=mutation)
    result=analyze_runs([run],tmp_path/'out',identity_path=identity)
    assert result['failures'][0]['minimum_signed_z_displacement_m']==-.02
    assert result['failures'][0]['diagnostic_predicate_mismatches']==0


def test_wrong_raw_success_or_step_counter_rejected(tmp_path):
    identity=frozen_identity(tmp_path)
    def mutation(rows):
        next(row for row in rows if row['kind']=='success_state')['capture']['step_numbers'][0]=99
    run=run_fixture(tmp_path,mutate=mutation)
    with pytest.raises(ValueError,match='step counter'):
        analyze_runs([run],tmp_path/'out',identity_path=identity)


def test_exclusive_output_and_no_causal_contact_claim(tmp_path):
    identity=frozen_identity(tmp_path);run=run_fixture(tmp_path)
    result=analyze_runs([run],tmp_path/'out',identity_path=identity)
    assert 'no contact, jam, force' in result['failures'][0]['interpretation']
    with pytest.raises(FileExistsError): analyze_runs([run],tmp_path/'out',identity_path=identity)
