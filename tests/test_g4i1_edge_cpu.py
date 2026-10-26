"""Original-source CPU clipping/EMA edge validation without simulator imports."""
import hashlib
import json

import pytest

from scripts import g4i1_edge_cpu as edge


def test_original_player_and_factory_edges_match_with_unbounded_reset_cache():
    result = edge.run_edge_check()
    assert result["passed"] and not result["gpu_used"]
    assert not result["actor_weights_loaded_or_mutated"]
    cases = {item["case"]: item for item in result["cases"]}
    assert set(cases) == {"normal", "action_clipping", "near_zero_rotation", "position_saturation"}
    clipped = cases["action_clipping"]
    assert clipped["reference_clipped"] == [[1., -1., 1., -1., 1., -1.]]
    # A mistaken second clamp of the EMA cache would erase this reset evidence.
    assert cases["normal"]["reference_ema"][0][0] > 1
    for item in cases.values():
        assert item["parity"]["comparison_count"] == 12
        assert item["parity"]["max_absolute_error"] == 0
        assert item["simulator_commands_submitted"] == 0
        assert item["synthetic_injected_mu"]
    json.dumps(result, allow_nan=False)


def test_fixture_detects_missing_clipping_and_incorrect_cache_reclipping(monkeypatch):
    monkeypatch.setattr(edge, "clip_policy_action", lambda mu: mu)
    result = edge.run_edge_check()
    clipping = next(item for item in result["cases"] if item["case"] == "action_clipping")
    assert not result["passed"] and not clipping["passed"]
    assert clipping["parity"]["stages"]["clipped"]["violation_count"] == 6


def test_fixture_detects_incorrect_ema_reclipping(monkeypatch):
    original = edge.smooth_action
    monkeypatch.setattr(edge, "smooth_action", lambda action, cache: original(action, cache).clamp(-1, 1))
    result = edge.run_edge_check()
    assert not result["passed"]
    assert any(not item["parity"]["stages"]["ema"]["passed"] for item in result["cases"])


def test_ast_extract_executes_only_requested_method(tmp_path):
    source = tmp_path / "fixture.py"
    source.write_text("raise RuntimeError('module must never execute')\n"
                      "class FactoryEnv:\n"
                      "    def _pre_physics_step(self, action):\n"
                      "        self.actions = action.clone()\n"
                      "    def unrelated(self):\n"
                      "        raise RuntimeError('must not extract')\n")
    method, identity = edge.extract_pre_step(source, hashlib.sha256(source.read_bytes()).hexdigest())
    assert callable(method)
    assert identity["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_wrong_source_sha_rejected_before_ast_parsing(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("raise RuntimeError('not trusted')\n")
    monkeypatch.setattr(edge.ast, "parse", lambda *a, **k: pytest.fail("AST parsed before identity passed"))
    with pytest.raises(ValueError, match="SHA mismatch"):
        edge.extract_pre_step(source, "0" * 64)


@pytest.mark.parametrize("text", ["class FactoryEnv: pass\n", "class Other: pass\n",
                                  "class FactoryEnv:\n    @staticmethod\n    def _pre_physics_step(self, action): pass\n"])
def test_missing_or_decorated_method_rejected(tmp_path, text):
    source = tmp_path / "bad_source.py"
    source.write_text(text)
    with pytest.raises(ValueError, match="one undecorated"):
        edge.extract_pre_step(source, hashlib.sha256(source.read_bytes()).hexdigest())


def test_cli_unique_output_and_existing_evidence_preserved(tmp_path, monkeypatch):
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(edge, "run_edge_check", lambda: {"passed": True})
    assert edge.main(["--output", str(output)]) == 0
    original = output.read_bytes()
    monkeypatch.setattr(edge, "run_edge_check", lambda: pytest.fail("old output was not rejected first"))
    with pytest.raises(FileExistsError):
        edge.main(["--output", str(output)])
    assert output.read_bytes() == original
