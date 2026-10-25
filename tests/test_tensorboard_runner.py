"""CPU-only tests for the actual Runner wrapper, without Torch/Isaac imports."""
from __future__ import annotations

import json

import pytest

from scripts.tensorboard_finalizer import wrap_runner_class


class FakeWriter:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def flush(self):
        self.calls.append("flush")
        if self.fail == "flush":
            raise OSError("simulated flush failure")

    def close(self):
        self.calls.append("close")
        if self.fail == "close":
            raise OSError("simulated close failure")


class FakeAgent:
    def __init__(self, writer, error=None, rank=0):
        self.writer = writer
        self.error = error
        self.global_rank = rank
        self.training_args = []

    def train(self, *args, **kwargs):
        self.training_args.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return "official-agent-return"


class FakeFactory:
    def __init__(self, agent):
        self.agent = agent
        self.create_args = []

    def create(self, *args, **kwargs):
        self.create_args.append((args, kwargs))
        return self.agent


def runner_class(agent, *, create=True):
    class FakeRunner:
        def __init__(self):
            self.algo_factory = FakeFactory(agent)

        def run_train(self, args):
            self.received_args = args
            if create:
                created = self.algo_factory.create("a2c_continuous", base_name="run", params={"unchanged": 1})
                return created.train()
            return "no-agent"

    return FakeRunner


def finalizer_payload(capsys):
    output = capsys.readouterr().out.splitlines()
    marker = next(line for line in output if line.startswith("FACTORY_TENSORBOARD_FINALIZER="))
    return json.loads(marker.split("=", 1)[1])


def test_runner_wrapper_preserves_original_return_arguments_and_factory(capsys):
    writer = FakeWriter()
    agent = FakeAgent(writer)
    cls = wrap_runner_class(runner_class(agent))
    runner = cls()
    original_factory = runner.algo_factory
    args = {"train": True, "checkpoint": None}

    assert runner.run_train(args) == "official-agent-return"
    assert runner.received_args is args
    assert runner.algo_factory is original_factory
    assert runner.algo_factory.create.__func__ is FakeFactory.create
    assert runner.algo_factory.create_args == [(("a2c_continuous",), {"base_name": "run", "params": {"unchanged": 1}})]
    assert agent.training_args == [((), {})]
    assert writer.calls == ["flush", "close"]
    payload = finalizer_payload(capsys)
    assert payload["status"] == "complete"
    assert payload["agents"][0]["success"] is True


def test_runner_wrapper_does_not_replace_training_exception(capsys):
    writer = FakeWriter()
    error = RuntimeError("original training error")
    cls = wrap_runner_class(runner_class(FakeAgent(writer, error=error)))
    runner = cls()

    with pytest.raises(RuntimeError, match="original training error") as captured:
        runner.run_train({"train": True})

    assert captured.value is error
    assert runner.algo_factory.create.__func__ is FakeFactory.create
    assert writer.calls == ["flush", "close"]
    payload = finalizer_payload(capsys)
    assert payload["status"] == "error"
    assert payload["training_error"]["type"] == "RuntimeError"


def test_runner_wrapper_records_writer_failure_even_when_training_returns(capsys):
    writer = FakeWriter(fail="flush")
    runner = wrap_runner_class(runner_class(FakeAgent(writer)))()

    assert runner.run_train({"train": True}) == "official-agent-return"

    payload = finalizer_payload(capsys)
    assert payload["status"] == "error"
    assert payload["training_error"] is None
    assert payload["agents"][0]["errors"][0]["operation"] == "flush"
    assert writer.calls == ["flush", "close"]


def test_rank_zero_missing_writer_cannot_report_complete(capsys):
    runner = wrap_runner_class(runner_class(FakeAgent(None)))()

    assert runner.run_train({"train": True}) == "official-agent-return"

    payload = finalizer_payload(capsys)
    assert payload["status"] == "error"
    assert payload["agents"][0]["writer_expected"] is True
    assert payload["agents"][0]["writer_present"] is False
    assert payload["agents"][0]["success"] is False


def test_nonzero_rank_without_writer_is_explicitly_not_expected(capsys):
    runner = wrap_runner_class(runner_class(FakeAgent(None, rank=1)))()

    runner.run_train({"train": True})

    payload = finalizer_payload(capsys)
    assert payload["status"] == "complete"
    assert payload["agents"][0]["writer_expected"] is False
    assert payload["agents"][0]["writer_present"] is False


def test_no_agent_created_cannot_report_complete(capsys):
    runner = wrap_runner_class(runner_class(FakeAgent(None), create=False))()

    assert runner.run_train({"train": True}) == "no-agent"

    payload = finalizer_payload(capsys)
    assert payload["status"] == "error"
    assert payload["no_agent_created"] is True


def test_runner_wrapper_is_idempotent():
    cls = runner_class(FakeAgent(FakeWriter()))
    assert wrap_runner_class(cls) is cls
    first = cls.run_train

    assert wrap_runner_class(cls) is cls
    assert cls.run_train is first


def test_runner_wrapper_persists_complete_json_and_no_temporary_file(tmp_path, monkeypatch):
    output = tmp_path / "finalizer.json"
    monkeypatch.setenv("FACTORY_G3A_FINALIZER_OUTPUT", str(output))
    runner = wrap_runner_class(runner_class(FakeAgent(FakeWriter())))()

    runner.run_train({"train": True})

    payload = json.loads(output.read_text())
    assert payload["status"] == "complete"
    assert payload["agents"][0]["flush_called"] is True
    assert payload["agents"][0]["close_called"] is True
    assert list(tmp_path.iterdir()) == [output]
