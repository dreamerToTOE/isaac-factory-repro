"""G3B read-only trajectory wrapper around the unchanged G2 evaluator.

Pass all ordinary evaluate_checkpoint.py arguments plus --trajectory-output
and --diagnostic-metadata. This delegates policy execution, normalization,
episode counting and recurrent resets to the original evaluator verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import runpy
import sys
from pathlib import Path

try:
    import success_metrics
    from evaluation_reporting import atomic_write_json
    from trajectory_capture import SCHEMA_VERSION, VectorTrajectoryRecorder, save_trajectory_npz, sha256_file
except ImportError:
    from scripts import success_metrics
    from scripts.evaluation_reporting import atomic_write_json
    from scripts.trajectory_capture import SCHEMA_VERSION, VectorTrajectoryRecorder, save_trajectory_npz, sha256_file


EXPECTED_CHECKPOINT_SHA256 = "f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f"


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--diagnostic-metadata", type=Path, required=True)
    wrapper_args, evaluator_args = parser.parse_known_args()
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--seed", type=int, required=True)
    check.add_argument("--max_steps", type=int, default=450)
    check.add_argument("--num_envs", type=int, default=32)
    check.add_argument("--episodes", type=int, default=3)
    check.add_argument("--task", required=True)
    check.add_argument("--stochastic", action="store_true")
    checked, _ = check.parse_known_args(evaluator_args)
    if checked.task != "Isaac-Factory-PegInsert-Direct-v0" or checked.stochastic:
        parser.error("G3B requires deterministic official PegInsert evaluation")
    if checked.num_envs != 32 or not 1 <= checked.episodes <= 3 or not 1 <= checked.max_steps <= 450:
        parser.error("G3B is bounded to 32 environments, <=3 episodes/env, <=450 policy steps")
    if checked.seed not in (1001, 1002, 1003):
        parser.error("G3B diagnostic seeds must be 1001, 1002 or 1003")
    if checked.max_steps < 149 * checked.episodes:
        parser.error("max_steps cannot complete the requested official 149-step episodes")
    outputs = [wrapper_args.trajectory_output, wrapper_args.diagnostic_metadata, checked.output]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        parser.error("diagnostic outputs must be distinct new paths")
    digest = sha256_file(checked.checkpoint)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        parser.error("checkpoint digest differs from the preserved G2 final checkpoint")

    original_capture = success_metrics.PreResetSuccessCapture
    state = {"recorders": [], "metadata": None, "arrays": None, "artifact": None}
    evaluator_path = Path(__file__).with_name("evaluate_checkpoint.py")

    def refresh_metadata():
        metadata = state["metadata"]
        if metadata is None:
            return
        if checked.output.is_file():
            result = json.loads(checked.output.read_text())
            metadata["evaluation_output"] = {"path": str(checked.output.resolve()),
                                             "sha256": sha256_file(checked.output),
                                             "status": result["status"],
                                             "aggregate": result["aggregate"],
                                             "episodes_completed": result["episodes_completed"]}
        atomic_write_json(wrapper_args.diagnostic_metadata, metadata)

    def persist_capture(capture_complete):
        metadata = state["metadata"]
        arrays = state["arrays"]
        metadata["status"] = "complete" if capture_complete else "incomplete_error"
        metadata["status_scope"] = "trajectory capture; evaluation_output.status separately records evaluator completion"
        metadata["sources"] = {path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                               for path in (Path(__file__), evaluator_path,
                                            Path(__file__).with_name("trajectory_capture.py"),
                                            Path(success_metrics.__file__))}
        metadata["policy_steps_recorded"] = int(arrays["policy_step"].shape[0])
        metadata["terminal_samples"] = int(arrays["done"].sum())
        if state["artifact"] is None:
            state["artifact"] = save_trajectory_npz(wrapper_args.trajectory_output, arrays, metadata)
        metadata["trajectory"] = state["artifact"]
        refresh_metadata()

    class DiagnosticCapture(original_capture):
        def __init__(self, env, convert=None, snapshot=None):
            self.recorder = VectorTrajectoryRecorder(env, checked.max_steps, checked.seed)
            state["recorders"].append(self.recorder)
            predicate = inspect.getsource(env._get_curr_successes)
            state["metadata"] = {
                "schema_version": SCHEMA_VERSION, "seed": checked.seed, "num_envs": env.num_envs,
                "checkpoint": str(checked.checkpoint.resolve()), "checkpoint_sha256": digest,
                "task": checked.task, "sampling_phase": "official success/engagement reward calls before automatic reset",
                "array_layout": "[global_policy_step, vector_env_index, optional_component]",
                "quaternion_order": "wxyz", "position_frame": "world (scene.env_origins added to task-local positions)",
                "official_error_frame": "held/target base world XY and world Z, as in official predicate",
                "axial_displacement_sign": "held_base.z - target_held_base.z; positive is above target",
                "policy_actions_semantics": "action supplied to official _pre_physics_step, after player/wrapper clipping, before EMA",
                "actions_semantics": "official EMA-filtered task action, dimensionless controller command",
                "controller_targets_semantics": "commanded pose; not measured contact",
                "contact_signals": {"available": False,
                                    "reason": "baseline exposes no configured measured contact sensor; no sensor/physics flag was enabled",
                                    "applied_wrench_used": False},
                "official_predicate_source_sha256": hashlib.sha256(predicate.encode()).hexdigest(),
                "official_success_threshold": float(env.cfg_task.success_threshold),
                "official_engage_threshold": float(env.cfg_task.engage_threshold),
                "official_xy_threshold_m": 0.0025,
                "official_z_threshold_m": float(env.cfg_task.fixed_asset_cfg.height * env.cfg_task.success_threshold),
                "fixed_asset_height_m": float(env.cfg_task.fixed_asset_cfg.height),
                "held_asset_height_m": float(env.cfg_task.held_asset_cfg.height),
                "step_dt_seconds": float(env.step_dt), "max_episode_length": int(env.max_episode_length),
                "expected_terminal_episode_step": int(env.max_episode_length) - 1,
                "official_rotation_check": False,
                "instrumentation": "detached GPU clones only; original evaluator loop and predicate return objects retained",
            }

            kwargs = {"snapshot": snapshot}
            if convert is not None:
                kwargs["convert"] = convert
            super().__init__(env, **kwargs)
            original_predicate = self.original
            recorder = self.recorder
            threshold = float(self.env.cfg_task.engage_threshold)

            def observed_predicate(success_threshold, check_rot=False):
                result = original_predicate(success_threshold=success_threshold, check_rot=check_rot)
                if float(success_threshold) == threshold and not check_rot:
                    recorder.record_engagement(result)
                elif float(success_threshold) == float(self.env.cfg_task.success_threshold) and not check_rot:
                    recorder.record_success(self.env, result)
                return result

            self.original_unobserved = original_predicate
            self.original = observed_predicate
            original_close = env.close

            def observed_close(*close_args, **close_kwargs):
                # The unchanged evaluator writes metrics before env.close.
                # Persist that linkage before AppLauncher can shut down Python.
                refresh_metadata()
                env.close = original_close
                return original_close(*close_args, **close_kwargs)

            env.close = observed_close

        def __enter__(self):
            super().__enter__()
            self.recorder.install_action_tap()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.recorder.restore_action_tap()
            super().__exit__(exc_type, exc_value, traceback)
            self.env._get_curr_successes = self.original_unobserved
            # Complete the only host transfer before env.close/AppLauncher
            # shutdown; CPU serialization later cannot depend on live Sim state.
            state["arrays"] = self.recorder.arrays()
            persist_capture(exc_type is None)

    success_metrics.PreResetSuccessCapture = DiagnosticCapture
    original_argv = sys.argv
    succeeded = False
    try:
        sys.argv = [str(evaluator_path), *evaluator_args]
        runpy.run_path(str(evaluator_path), run_name="__main__")
        succeeded = True
    finally:
        sys.argv = original_argv
        success_metrics.PreResetSuccessCapture = original_capture
        if state["recorders"]:
            if state["artifact"] is None:
                state["arrays"] = state["recorders"][0].arrays()
                persist_capture(succeeded)
            refresh_metadata()
            print("FACTORY_DIAGNOSTIC_OUTPUT=" + str(wrapper_args.diagnostic_metadata.resolve()), flush=True)


if __name__ == "__main__":
    main()
