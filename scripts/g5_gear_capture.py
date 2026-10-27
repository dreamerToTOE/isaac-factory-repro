"""Read-only trace tap for completed automatic reset calls."""
from types import MethodType


class ResetCapture:
    """Delegate the original reset and record its IDs only after success."""
    def __init__(self, env, convert):
        self.env = env
        self.convert = convert
        self.original = env._reset_idx
        self.events = []

    def __enter__(self):
        tap = self
        def wrapped(instance, env_ids):
            result = tap.original(env_ids)
            tap.events.append(tap.convert(env_ids))
            return result
        self.env._reset_idx = MethodType(wrapped, self.env)
        return self

    def begin_decision(self):
        self.events.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self.env._reset_idx = self.original
