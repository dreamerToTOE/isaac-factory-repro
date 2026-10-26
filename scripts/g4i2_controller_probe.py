"""Read-only diagnostics around the pinned Factory torque function.

The probe executes the original code object exactly once with its original
arguments. A private globals dictionary replaces only ``torch`` with a proxy;
no upstream source, module globals, global torch function or controller math is
changed. Captures are diagnostic requested/software-clipped values, not measured
actuator effort or contact force. CPU mocks exercise this module without Isaac.
"""
from dataclasses import dataclass
from types import FunctionType

import torch


def _finite_tensor(value, name):
    if not torch.is_tensor(value) or value.layout != torch.strided:
        raise ValueError(f"{name} must be a strided Tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"non-finite {name}")


@dataclass(frozen=True)
class TorqueCapture:
    requested_torque: torch.Tensor
    postclip_torque: torch.Tensor
    task_wrench: torch.Tensor
    clamp_min: float = -100.0
    clamp_max: float = 100.0

    def as_dict(self):
        """Tensor fields remain on their original device until caller serializes."""
        return {"requested_torque": self.requested_torque,
                "postclip_torque": self.postclip_torque,
                "task_wrench": self.task_wrench,
                "clamp_min": self.clamp_min, "clamp_max": self.clamp_max}


class _TorchProxy:
    def __init__(self, original, owner):
        self._original = original
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._original, name)

    def clamp(self, input, *args, **kwargs):
        owner = self._owner
        owner._clamp_count += 1
        if owner._clamp_count != 1:
            raise RuntimeError("controller probe expected exactly one torque clamp")
        # Pinned factory_control.py calls this exact scalar-bound, out-of-place
        # clamp. Any source/API change is an instrumentation failure, not a
        # reason to silently reinterpret a different clamp as torque clipping.
        if args or set(kwargs) != {"min", "max"} or kwargs["min"] != -100.0 or kwargs["max"] != 100.0:
            raise RuntimeError("controller torque clamp signature changed")
        _finite_tensor(input, "requested torque")
        owner._requested = input.detach().clone()
        result = self._original.clamp(input, *args, **kwargs)
        owner._clamped = result
        return result


class TorqueProbe:
    """Callable temporary wrapper for ``fc.compute_dof_torque``.

    ``last`` is cleared before each call and remains None after any failure.
    Only the existing controller computes outputs; the probe never recomputes
    torque or submits a command. This wrapper is intentionally non-reentrant.
    """
    def __init__(self, function):
        if not isinstance(function, FunctionType):
            raise TypeError("controller probe requires the original Python function")
        original_torch = function.__globals__.get("torch")
        if original_torch is not torch:
            raise ValueError("controller must use its unmodified torch module global")
        self.original = function
        self.last = None
        self.call_count = 0
        self._active = False
        self._clamp_count = 0
        self._requested = self._clamped = None
        proxy = _TorchProxy(original_torch, self)
        self.probed = FunctionType(function.__code__, {**function.__globals__, "torch": proxy},
                                   function.__name__, function.__defaults__, function.__closure__)
        self.probed.__kwdefaults__ = function.__kwdefaults__
        self.probed.__annotations__ = function.__annotations__.copy()
        self.probed.__doc__ = function.__doc__

    def __call__(self, *args, **kwargs):
        if self._active:
            raise RuntimeError("controller probe does not support reentrant calls")
        self.last = None
        self._clamp_count = 0
        self._requested = self._clamped = None
        self._active = True
        self.call_count += 1
        try:
            result = self.probed(*args, **kwargs)
            if self._clamp_count != 1:
                raise RuntimeError("controller probe expected exactly one torque clamp")
            if not isinstance(result, tuple) or len(result) != 2 or result[0] is not self._clamped:
                raise RuntimeError("controller return does not match captured torque clamp")
            torque, wrench = result
            _finite_tensor(torque, "postclip torque")
            _finite_tensor(wrench, "task wrench")
            self.last = TorqueCapture(self._requested, torque.detach().clone(), wrench.detach().clone())
            return result
        finally:
            self._active = False
            self._requested = self._clamped = None


def position_constraint_activity(relative_target, lower, upper):
    """Observe Cartesian bounds without applying or changing any constraint.

    ``relative_target`` is the original controller's pre-clip target relative
    to its action frame, in meters; lower/upper are its original bounds.
    Strict outside flags mean clipping changes the value. Boundary flags use
    exact comparisons, adding no new tolerance or controller threshold.
    """
    _finite_tensor(relative_target, "relative position target")
    if relative_target.ndim != 2 or relative_target.shape[1] != 3:
        raise ValueError("relative position target must have shape [batch, 3]")
    low = torch.as_tensor(lower, device=relative_target.device, dtype=relative_target.dtype)
    high = torch.as_tensor(upper, device=relative_target.device, dtype=relative_target.dtype)
    _finite_tensor(low, "lower position bound")
    _finite_tensor(high, "upper position bound")
    if low.ndim > 1 or high.ndim > 1 or low.numel() not in (1, 3) or high.numel() not in (1, 3):
        raise ValueError("position bounds must be scalar or length 3")
    if not (low < high).all().item():
        raise ValueError("lower position bound must be below upper")
    return {"below_lower": (relative_target < low).detach().clone(),
            "above_upper": (relative_target > high).detach().clone(),
            "at_or_beyond_lower": (relative_target <= low).detach().clone(),
            "at_or_beyond_upper": (relative_target >= high).detach().clone(),
            "would_clip": ((relative_target < low) | (relative_target > high)).detach().clone()}
