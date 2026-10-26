"""Independent CPU functions exercise instrumentation; no simulator is imported."""
import pytest
import torch

from scripts.g4i2_controller_probe import TorqueProbe, position_constraint_activity


CALLS = []


def controller(value, *, offset=0):
    CALLS.append((value, value.stride()))
    requested = value + offset
    wrench = requested[:, :6] * 2
    clipped = torch.clamp(requested, min=-100.0, max=100.0)
    return clipped, wrench


def test_exact_original_code_once_no_input_or_global_mutation():
    CALLS.clear()
    original_globals = controller.__globals__.copy()
    original_clamp = torch.clamp
    value = torch.tensor([[-150., 0., 150.], [-90., 30., 120.]]).T
    before = value.clone()
    probe = TorqueProbe(controller)
    output = probe(value, offset=1)
    assert probe.probed.__code__ is controller.__code__
    assert len(CALLS) == 1 and CALLS[0][0] is value and CALLS[0][1] == value.stride()
    assert probe.call_count == 1
    assert torch.equal(value, before)
    assert torch.clamp is original_clamp
    assert all(controller.__globals__[key] is val for key, val in original_globals.items())
    assert probe.probed.__globals__["torch"] is not torch
    assert output[0].shape == value.shape
    assert torch.equal(output[0], torch.tensor([[-100., -89.], [1., 31.], [100., 100.]]))
    assert torch.equal(probe.last.requested_torque, before + 1)
    assert torch.equal(probe.last.postclip_torque, output[0])
    assert torch.equal(probe.last.task_wrench, output[1])
    assert probe.last.postclip_torque is not output[0]
    output[0].zero_()
    assert probe.last.postclip_torque[0, 0] == -100


def test_original_defaults_keyword_defaults_and_closure_preserved():
    multiplier = 3
    def local(value, offset=2, *, additive=4):
        desired = value * multiplier + offset + additive
        return torch.clamp(desired, min=-100., max=100.), desired[:, :6]
    probe = TorqueProbe(local)
    actual = probe(torch.ones(1, 9))
    assert torch.equal(actual[0], torch.full((1, 9), 9.))
    assert probe.probed.__closure__ is local.__closure__
    assert probe.probed.__defaults__ is local.__defaults__
    assert probe.probed.__kwdefaults__ is local.__kwdefaults__


def test_original_return_tensors_retained_not_recomputed():
    outputs = []
    def original(value):
        out = (torch.clamp(value, min=-100., max=100.), value[:, :6])
        outputs.append(out)
        return out
    probe = TorqueProbe(original)
    actual = probe(torch.zeros(1, 9))
    assert actual is outputs[0]


def test_proxy_delegates_other_torch_operations_unchanged():
    def original(value):
        desired = torch.zeros_like(value) + torch.sin(value)
        return torch.clamp(desired, min=-100., max=100.), torch.cat([value, value], dim=1)
    actual = TorqueProbe(original)(torch.tensor([[0., torch.pi/2]]))
    assert torch.equal(actual[0], torch.tensor([[0., 1.]]))


@pytest.mark.parametrize("kind", ["missing", "multiple", "wrong_bounds", "positional", "different_return"])
def test_changed_clamp_contract_fails_without_capture(kind):
    def original(value):
        if kind == "missing": return value, value
        if kind == "wrong_bounds": return torch.clamp(value, min=-10., max=10.), value
        if kind == "positional": return torch.clamp(value, -100., 100.), value
        out = torch.clamp(value, min=-100., max=100.)
        if kind == "multiple": out = torch.clamp(out, min=-100., max=100.)
        if kind == "different_return": out = out.clone()
        return out, value
    probe = TorqueProbe(original)
    with pytest.raises(RuntimeError): probe(torch.ones(1, 9))
    assert probe.last is None and not probe._active


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_requested_torque_rejected_before_clamp(bad):
    probe = TorqueProbe(controller)
    with pytest.raises(ValueError, match="non-finite requested torque"):
        probe(torch.tensor([[bad, 1.]]))
    assert probe.last is None


def test_nonfinite_wrench_rejected_and_previous_capture_cleared():
    def original(value):
        return torch.clamp(value, min=-100., max=100.), value / 0
    probe = TorqueProbe(original)
    with pytest.raises(ValueError, match="non-finite task wrench"): probe(torch.ones(1, 9))
    assert probe.last is None
    normal = TorqueProbe(controller)
    normal(torch.ones(1, 9))
    assert normal.last is not None
    with pytest.raises(ValueError): normal(torch.full((1, 9), float("nan")))
    assert normal.last is None


def test_original_exception_propagates_and_next_call_works():
    def original(value):
        if value is None: raise LookupError("controller rejected")
        return torch.clamp(value, min=-100., max=100.), value
    probe = TorqueProbe(original)
    with pytest.raises(LookupError): probe(None)
    assert probe.last is None
    probe(torch.ones(1, 9))
    assert probe.last is not None and probe.call_count == 2


@pytest.mark.parametrize("function", [None, torch.clamp, object()])
def test_only_original_python_function_accepted(function):
    with pytest.raises(TypeError): TorqueProbe(function)


def test_as_dict_reports_requested_software_clipped_and_wrench_only():
    probe = TorqueProbe(controller)
    probe(torch.tensor([[101., -101., 100., -100., 0., 1.]]))
    record = probe.last.as_dict()
    assert set(record) == {"requested_torque", "postclip_torque", "task_wrench", "clamp_min", "clamp_max"}
    assert record["clamp_min"] == -100 and record["clamp_max"] == 100


def test_position_activity_distinguishes_touching_bound_and_actual_clipping():
    value = torch.tensor([[-.041, -.04, .04], [.041, 0., -.039]])
    before = value.clone()
    flags = position_constraint_activity(value, -.04, .04)
    assert torch.equal(flags["would_clip"], torch.tensor([[True, False, False], [True, False, False]]))
    assert torch.equal(flags["at_or_beyond_lower"], torch.tensor([[True, True, False], [False, False, False]]))
    assert torch.equal(flags["at_or_beyond_upper"], torch.tensor([[False, False, True], [True, False, False]]))
    assert torch.equal(value, before)


def test_position_activity_supports_per_dimension_bounds():
    flags = position_constraint_activity(torch.tensor([[-2., 2., 0.]]), [-1., -2., -3.], [1., 2., 3.])
    assert flags["would_clip"].tolist() == [[True, False, False]]


@pytest.mark.parametrize("value,low,high", [
    (torch.ones(3), -1, 1), (torch.ones(1, 4), -1, 1),
    (torch.full((1, 3), float("nan")), -1, 1),
    (torch.ones(1, 3), 1, 1), (torch.ones(1, 3), 2, 1),
    (torch.ones(1, 3), [0., 1.], 2), (torch.ones(1, 3), -1, float("inf")),
])
def test_position_activity_invalid_input_rejected(value, low, high):
    with pytest.raises(ValueError): position_constraint_activity(value, low, high)
