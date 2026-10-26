"""Synthetic CPU controls for streaming episode statistics, not simulator data."""
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.g4i2_episode_statistics import process_runs, process_seed


def fixture(tmp_path, *, seed=1001, mutate=None):
    run=tmp_path/f"run{seed}"
    run.mkdir()
    trace=run/"i2_trace.jsonl"
    rows=[]
    for step in range(1,4):
        raw=[[.5]*6,[2.]*6]
        rows.append(dict(kind="action",policy_step=step,warmup=step<=2,raw_mu=raw,
                         clipped=[[.5]*6,[1.]*6],ema=[[.1]*6,[.2]*6],
                         physical_position_increment_m=[[.002]*3,[.004]*3],
                         physical_rotation_increment_rad=[[.0097]*3,[.0194]*3],
                         observation=[[step]*19,[100+step]*19],normalized=[[0.]*19]*2))
        for substep in (1,2):
            rows.append(dict(kind="controller",policy_step=step,substep=substep,
                             target_position=[[0.,0.,1.],[1.,0.,1.]],
                             target_quaternion=[[1.,0.,0.,0.]]*2,
                             requested_torque=[[1.]*7,[100.]*7],software_torque=[[1.]*7,[87.]*7],
                             effort_limits=[[87.]*7]*2,position_constraint_active=[[False]*3,[True]*3],
                             orientation_constraint_active=[[True,True,False]]*2,applied_sim_effort="UNKNOWN"))
        rows.append(dict(kind="success_state",policy_step=step,capture=dict(success=[step>=2,False],step_numbers=[step,step],
                         geometric_snapshots=dict(held_base_pos=[[0.,0.,0.]]*2,target_held_base_pos=[[0.,0.,0.]]*2))))
        rows.append(dict(kind="timing",policy_step=step,warmup=step<=2,timings_ms=dict(forward=step,observation_to_command=step+1),
                         wall_command_monotonic_ns=step*70_000_000,control_loop_ms=70,sim_dt_s=1/15,
                         event_sections=[dict(stage="forward",device_ms=step/10,succeeded=True)]))
    if mutate: mutate(rows)
    trace.write_text("".join(json.dumps(row)+"\n" for row in rows))
    episodes=[dict(evaluation_seed=seed,env_id=i,episode_id=0,steps=3,
                   first_success_step=2 if i==0 else None,ever_success=i==0,terminal_success=i==0,
                   first_success_time_seconds=2/15 if i==0 else None,termination_reason="timeout",
                   initial_randomized_configuration=dict(provenance="synthetic_test_only",x=i),
                   pre_reset_terminal_state=dict(x=i),trace=dict(path=str(trace),first_policy_step=1,last_policy_step=3)) for i in range(2)]
    path=run/"episodes.json"
    path.write_text(json.dumps(episodes))
    return run,rows,episodes


def test_streaming_single_env_selection_and_scientific_warmup_included(tmp_path):
    run,rows,episodes=fixture(tmp_path)
    result=process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)
    assert result["coverage_status"] == "COMPLETE"
    first,second=result["episodes"]
    assert first["statistics"]["actions"]["dimensions"][0]["raw_outside_unit_rate"] == 0
    assert second["statistics"]["actions"]["dimensions"][0]["raw_outside_unit_rate"] == 1
    assert first["statistics"]["controller"]["joints"][0]["requested_torque"]["mean"] == 1
    assert second["statistics"]["controller"]["joints"][0]["requested_torque"]["mean"] == 100
    stats=first["statistics"]
    assert stats["original_warmup_steps"] == 2
    assert stats["timing"]["stages_ms"]["forward"]["n"] == 3
    assert stats["timing"]["command_intervals_ms"]["n"] == 2
    assert stats["coverage"]["controller_records"] == 6
    assert stats["device_sections_ms"]["forward"]["mean"] == pytest.approx(.2)
    assert first["initial_randomized_configuration"] == episodes[0]["initial_randomized_configuration"]
    assert result["source_trace_sha256"] == hashlib.sha256((run/"i2_trace.jsonl").read_bytes()).hexdigest()
    assert (tmp_path/"out"/"episodes_with_statistics.csv").exists()


def test_failure_external_reference_includes_all_step_data_and_no_new_trace(tmp_path):
    run,rows,episodes=fixture(tmp_path)
    result=process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)
    failure=result["episodes"][1]["raw_trace_reference"]
    assert failure["failure_episode"] and failure["env_id"] == 1
    raw=(run/"i2_trace.jsonl").read_bytes()
    selected=raw[failure["byte_range_start_inclusive"]:failure["byte_range_end_exclusive"]]
    selected_rows=[json.loads(line) for line in selected.splitlines()]
    assert sum(row["kind"] == "action" for row in selected_rows) == 3
    assert sum(row["kind"] == "success_state" for row in selected_rows) == 3
    assert selected_rows[0]["observation"][1] == [101]*19
    assert set(path.name for path in (tmp_path/"out").iterdir()) == {"episodes_with_statistics.json","episodes_with_statistics.csv","coverage.json"}
    assert not list(tmp_path.glob("g4i2_episode_statistics_*"))


def test_csv_contains_real_per_episode_statistics_and_original_state(tmp_path):
    run,_,_=fixture(tmp_path)
    process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)
    with (tmp_path/"out"/"episodes_with_statistics.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert rows[1]["failure_category"] == "never_success"
    assert json.loads(rows[1]["initial_randomized_configuration"])["x"] == 1
    assert json.loads(rows[1]["action_statistics"])["samples"] == 3
    assert json.loads(rows[1]["controller_statistics"])["samples"] == 6


@pytest.mark.parametrize("kind",["action","controller","timing","success_state"])
def test_missing_required_step_rejects_before_output_creation(tmp_path,kind):
    run,_,_=fixture(tmp_path,mutate=lambda rows:rows.__setitem__(slice(None),[r for r in rows if not(r["kind"]==kind and r["policy_step"]==2)]))
    with pytest.raises(ValueError,match="incomplete episode"):
        process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("kind",["action","controller","timing","success_state"])
def test_duplicate_required_records_rejected(tmp_path,kind):
    def mutation(rows):
        index=next(i for i,row in enumerate(rows) if row["kind"]==kind and row["policy_step"]==2)
        rows.insert(index,deepcopy(rows[index]))
    run,_,_=fixture(tmp_path,mutate=mutation)
    with pytest.raises(ValueError):
        process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)


def test_incomplete_mode_remains_explicit(tmp_path):
    run,_,_=fixture(tmp_path,mutate=lambda rows:rows.__setitem__(slice(None),[r for r in rows if not(r["kind"]=="controller" and r["policy_step"]==2)]))
    result=process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2,strict=False)
    assert result["coverage_status"] == "INCOMPLETE"
    assert "MISSING_CONTROLLER_STEPS" in str(result["episodes"][0]["statistics"]["coverage"]["errors"])


def test_success_trace_rechecked_against_episode_flags(tmp_path):
    def mutation(rows):
        next(row for row in rows if row["kind"]=="success_state" and row["policy_step"]==3)["capture"]["success"][0]=False
    run,_,_=fixture(tmp_path,mutate=mutation)
    with pytest.raises(ValueError,match="RAW_SUCCESS_FLAGS_DISAGREE"):
        process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)


def test_wrong_episode_step_and_missing_observations_rejected(tmp_path):
    def mutation(rows):
        next(row for row in rows if row["kind"]=="action").pop("observation")
        next(row for row in rows if row["kind"]=="success_state")["capture"]["step_numbers"][0]=99
    run,_,_=fixture(tmp_path,mutate=mutation)
    with pytest.raises(ValueError,match="RAW_EPISODE_STEP_COUNTERS_DISAGREE"):
        process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)


def test_output_exclusive_and_scratch_spools_cleaned(tmp_path):
    run,_,_=fixture(tmp_path)
    output=tmp_path/"out";scratch=tmp_path/"scratch";scratch.mkdir()
    process_seed(run/"i2_trace.jsonl",run/"episodes.json",output,expected_controller_substeps=2,scratch_dir=scratch)
    assert not list(scratch.iterdir())
    with pytest.raises(FileExistsError): process_seed(run/"i2_trace.jsonl",run/"episodes.json",output,expected_controller_substeps=2)


def test_multiseed_cli_api_processes_seeds_without_pairing_claim(tmp_path):
    one,_,_=fixture(tmp_path,seed=1001)
    two,_,_=fixture(tmp_path,seed=1002)
    result=process_runs([one,two],tmp_path/"combined",expected_controller_substeps=2)
    assert len(result["episodes"]) == 4
    assert result["aggregate"]["episodes_completed"] == 4
    assert result["aggregate"]["terminal_success_rate"] == .5
    assert result["aggregate"]["archived_comparison"]["kind"] == "descriptive_unpaired"


def test_duplicate_multiseed_run_rejected(tmp_path):
    run,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match="duplicate episode identity across"):
        process_runs([run,run],tmp_path/"combined",expected_controller_substeps=2)


def test_invalid_matrix_and_corrupt_json_do_not_create_results(tmp_path):
    run,_,_=fixture(tmp_path,mutate=lambda rows:rows[0].update(raw_mu=[[0.]*5]*2))
    with pytest.raises(ValueError,match="width"):
        process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)
    (run/"i2_trace.jsonl").write_text('{bad json\n')
    with pytest.raises(ValueError,match="invalid JSON"):
        process_seed(run/"i2_trace.jsonl",run/"episodes.json",tmp_path/"out",expected_controller_substeps=2)
