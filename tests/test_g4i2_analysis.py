"""Deterministic synthetic CPU controls for I2 reporting, never simulated data."""
import json
from copy import deepcopy

import numpy as np
import pytest

from scripts.g4i2_analysis import (POLICY_INTERVAL_MS, classify_episode, plot_distributions,
                                  statistics, summarize_actions, summarize_controller,
                                  summarize_episodes, summarize_timings, write_episode_csv)


def timing(step, ms=5, warmup=False):
    return dict(kind="timing", policy_step=step, warmup=warmup,
                timings_ms={"forward": ms, "observation_to_command": ms+2},
                wall_command_monotonic_ns=step*70_000_000, control_loop_ms=70, sim_dt_s=1/15)


def action():
    raw = np.array([[-2., -1., 0., .5, 1., 2.], [0., 0., 0., 0., 0., 0.]])
    return dict(kind="action", raw_mu=raw.tolist(), clipped=np.clip(raw,-1,1).tolist(),
                ema=(raw*.2).tolist(), physical_position_increment_m=(raw[:,:3]*.004).tolist(),
                physical_rotation_increment_rad=(raw[:,3:]*.0194).tolist())


def controller():
    requested = [[100., 1., 2., 3., 4., 5., 6.], [-100., 0., 0., 0., 0., 0., 0.]]
    software = [[87., 1., 2., 3., 4., 5., 6.], [-87., 0., 0., 0., 0., 0., 0.]]
    return dict(kind="controller", target_position=[[0.,0.,1.],[0.,0.,1.]],
                target_quaternion=[[1.,0.,0.,0.],[1.,0.,0.,0.]], requested_torque=requested,
                software_torque=software, effort_limits=[[87.]*7]*2,
                position_constraint_active=[[True,False,False],[False,False,True]])


def test_statistics_exact_quantiles_and_population_std():
    result=statistics([1,2,3,4])
    assert result == pytest.approx(dict(n=4,mean=2.5,p50=2.5,p95=3.85,p99=3.97,min=1,max=4,std=np.sqrt(1.25)))
    assert statistics([])["mean"] is None


@pytest.mark.parametrize("bad", [[1,float('nan')],[float('inf')],[[1,2]]])
def test_nonfinite_or_multidimensional_statistics_rejected(bad):
    with pytest.raises(ValueError): statistics(bad)


def test_warmup_excluded_clock_frequency_and_rtf_are_not_forward_inverse():
    rows=[timing(0,99,True),timing(1,1),timing(2,3),timing(3,5)]
    result=summarize_timings(rows)
    assert result["stages_ms"]["forward"]["n"] == 3
    assert result["stages_ms"]["forward"]["mean"] == 3
    assert result["warmup_records"] == 1
    assert result["command_intervals_ms"]["n"] == 2
    assert result["command_deadline_misses"] == dict(count=2,n=2,rate=1.)
    assert result["wall_clock_command_frequency_hz"] == pytest.approx(1000/70)
    assert result["real_time_factor"] == pytest.approx((1/15)/.07)
    assert result["observation_to_command_deadline_misses"]["count"] == 0


def test_deadline_boundary_is_strict_and_has_correct_denominator():
    rows=[timing(0),timing(1),timing(2)]
    rows[0]["timings_ms"]["observation_to_command"]=POLICY_INTERVAL_MS
    rows[1]["timings_ms"]["observation_to_command"]=POLICY_INTERVAL_MS+.1
    result=summarize_timings(rows)
    assert result["observation_to_command_deadline_misses"] == dict(count=1,n=3,rate=1/3)


def test_missing_clock_is_unknown_and_no_gap_bridging():
    rows=[timing(0),timing(1),timing(2)]
    rows[1].pop("wall_command_monotonic_ns")
    result=summarize_timings(rows)
    assert result["command_intervals_ms"]["n"] == 0
    assert result["wall_clock_command_frequency_hz"] is None
    assert result["command_deadline_misses"]["rate"] is None
    assert summarize_timings([])["real_time_factor"] is None


@pytest.mark.parametrize("mutation", [lambda r:r.update(warmup="no"),
    lambda r:r["timings_ms"].update(forward=-1), lambda r:r["timings_ms"].update(forward=float('nan')),
    lambda r:r.update(wall_command_monotonic_ns=1.1), lambda r:r.update(control_loop_ms=-1),
    lambda r:r.update(sim_dt_s=float('inf'))])
def test_invalid_timing_values_rejected(mutation):
    row=timing(1); mutation(row)
    with pytest.raises(ValueError): summarize_timings([row])


@pytest.mark.parametrize("second", [0,1])
def test_duplicate_or_out_of_order_timestamps_rejected(second):
    with pytest.raises(ValueError): summarize_timings([timing(1),timing(second)])


def test_action_boundary_saturation_is_distinct_from_clipping():
    result=summarize_actions([action()])
    assert result["samples"] == 2
    assert result["dimensions"][0]["raw_outside_unit_rate"] == .5
    assert result["dimensions"][1]["raw_outside_unit_rate"] == 0
    assert result["dimensions"][1]["clipped_at_unit_rate"] == .5
    assert result["dimensions"][2]["clipped_at_unit_rate"] == 0
    assert result["dimensions"][0]["physical_unit"] == "m"
    assert result["dimensions"][4]["physical_unit"] == "rad"


def test_analysis_never_adds_ema_clipping():
    row=action(); row["ema"][0][0]=2
    result=summarize_actions([row])
    assert result["dimensions"][0]["ema"]["max"] == 2
    assert row["ema"][0][0] == 2


@pytest.mark.parametrize("key,value", [("raw_mu",[[0.]*5]),("clipped",[[0.]*6]),
    ("physical_position_increment_m",[[0.,0.,float('nan')]])])
def test_broken_action_chain_rejected(key,value):
    row=action(); row[key]=value
    with pytest.raises(ValueError): summarize_actions([row])


def test_inconsistent_clip_rejected():
    row=action(); row["clipped"][0][0]=-.9
    with pytest.raises(ValueError): summarize_actions([row])


def test_controller_separates_requested_software_configured_and_unknown_applied():
    result=summarize_controller([controller()])
    assert result["joints"][0]["software_changed_rate"] == 1
    assert result["joints"][0]["requested_above_configured_limit_count"] == 2
    assert result["joints"][0]["software_at_or_above_configured_limit_count"] == 2
    assert result["applied_sim_effort"] == "UNKNOWN"
    assert result["constraints"]["position_constraint_active"][0]["rate"] == .5
    assert result["constraints"]["orientation_constraint_active"] == "UNKNOWN"


def test_controller_optional_diagnostics_are_reported_without_inference():
    row=controller(); row["applied_sim_effort"]=[[1.]*7]*2
    row["task_space_pd_output"]=[[2.]*6]*2
    result=summarize_controller([row])
    assert result["applied_sim_effort"][0]["mean"] == 1
    assert result["task_space_pd_output"][0]["mean"] == 2


@pytest.mark.parametrize("key,value", [("effort_limits",[[0.]*7]*2),
    ("requested_torque",[[1.]*6]*2),("target_quaternion",[[.1,0,0,0]]*2),
    ("position_constraint_active",[[2,0,0]]*2)])
def test_invalid_controller_diagnostics_rejected(key,value):
    row=controller(); row[key]=value
    with pytest.raises(ValueError): summarize_controller([row])


def episode(ever,terminal,index=0):
    return dict(evaluation_seed=1001,episode_id=index,ever_success=ever,terminal_success=terminal,
                first_success_time_seconds=2. if ever else None,
                pre_reset_terminal_state={"position_error_m":.001},geometric_errors={"xy_error_m":.001})


def test_failure_classification_distinguishes_never_and_lost_success():
    never=classify_episode(episode(False,False))
    lost=classify_episode(episode(True,False))
    assert never["category"] == "never_success"
    assert lost["category"] == "lost_success_before_terminal"
    assert lost["recorded_geometric_errors"] == {"xy_error_m":.001}
    assert "causal" in lost["interpretation"]


def test_descriptive_unpaired_comparison_retains_denominator_and_ever_terminal():
    result=summarize_episodes([episode(True,True,0),episode(True,False,1),episode(False,False,2)])
    assert result["ever_success_rate"] == 2/3
    assert result["terminal_success_rate"] == 1/3
    assert result["archived_comparison"]["archived_rate"] == 280/288
    assert result["archived_comparison"]["kind"] == "descriptive_unpaired"
    assert result["failure_categories"] == dict(terminal_success=1,lost_success_before_terminal=1,never_success=1)
    assert len(result["terminal_success_wilson95"]) == 2
    assert summarize_episodes([])["terminal_success_rate"] is None


def test_pairing_cannot_be_silently_asserted():
    with pytest.raises(ValueError): summarize_episodes([],pairing_verified=True)


def test_duplicate_episodes_and_invalid_success_accounting_rejected():
    with pytest.raises(ValueError): summarize_episodes([episode(True,True),episode(True,True)])
    with pytest.raises(ValueError): classify_episode(episode(False,True))
    row=episode(False,False); row["first_success_time_seconds"]=1
    with pytest.raises(ValueError): classify_episode(row)


def test_plots_are_real_files_and_cannot_overwrite(tmp_path):
    pytest.importorskip("matplotlib")
    records=[timing(0),timing(1),action()]
    paths=plot_distributions(records,tmp_path)
    assert len(paths) == 4
    assert all(__import__('pathlib').Path(path).stat().st_size > 100 for path in paths)
    with pytest.raises(FileExistsError): plot_distributions(records,tmp_path)


def test_no_input_mutation_and_json_has_no_nan():
    rows=[timing(0),timing(1)]
    original=deepcopy(rows)
    result=summarize_timings(rows)
    assert rows == original
    json.dumps(result,allow_nan=False)


def test_episode_csv_retains_pre_reset_evidence_and_refuses_overwrite(tmp_path):
    import csv
    path=tmp_path/"episodes.csv"
    write_episode_csv([episode(True,False)], path)
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    assert rows[0]["failure_category"] == "lost_success_before_terminal"
    assert json.loads(rows[0]["pre_reset_terminal_state"])["position_error_m"] == .001
    assert rows[0]["initial_randomized_configuration"] == '"UNKNOWN"'
    with pytest.raises(FileExistsError): write_episode_csv([],path)


def test_record_alignment_cannot_be_hidden_by_cross_record_counts():
    first,second=action(),action()
    first["ema"]=first["ema"][:1]
    second["ema"] += [second["ema"][0]]
    with pytest.raises(ValueError): summarize_actions([first,second])
    first,second=controller(),controller()
    first["target_position"]=first["target_position"][:1]
    second["target_position"] += [second["target_position"][0]]
    with pytest.raises(ValueError): summarize_controller([first,second])
