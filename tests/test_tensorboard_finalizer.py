from __future__ import annotations

from scripts.tensorboard_finalizer import finalize_writer
from scripts.tensorboard_tail import validate_tail


class Writer:
    def __init__(self, fail=None): self.calls=[]; self.fail=fail
    def flush(self):
        self.calls.append("flush")
        if self.fail == "flush": raise RuntimeError("flush")
    def close(self):
        self.calls.append("close")
        if self.fail == "close": raise RuntimeError("close")


def test_finalizer_flushes_before_close():
    writer=Writer(); result=finalize_writer(writer)
    assert writer.calls == ["flush", "close"] and result["success"]


def test_finalizer_records_writer_failure_without_hiding_it():
    writer=Writer("flush"); result=finalize_writer(writer)
    assert writer.calls == ["flush", "close"] and not result["success"] and result["errors"]


def test_tensorboard_tail_accepts_complete_finite_tail():
    points=[{"tag":"losses/a_loss","step":3,"epoch":3,"value":1.0},
            {"tag":"rewards/iter","step":3,"epoch":3,"value":2.0}]
    assert validate_tail(points,3)["status"] == "complete"


def test_tensorboard_tail_rejects_missing_tail_without_backfill():
    points=[{"tag":"losses/a_loss","step":2,"epoch":2,"value":1.0},
            {"tag":"rewards/iter","step":3,"epoch":3,"value":2.0}]
    result=validate_tail(points,3)
    assert result["status"] == "incomplete" and "losses/a_loss" in result["missing_tags_at_expected_epoch"]


def test_tensorboard_tail_rejects_infinite_value_and_explicit_missing_tag():
    points = [{"tag": "losses/a_loss", "step": 3, "epoch": 3, "value": float("inf")}]
    result = validate_tail(points, 3, expected_tags=["losses/a_loss", "rewards/iter"])
    assert result["status"] == "incomplete"
    assert result["nonfinite_count"] == 1
    assert "rewards/iter" in result["missing_tags_at_expected_epoch"]
