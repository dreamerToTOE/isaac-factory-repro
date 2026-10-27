#!/usr/bin/env python3
"""R2-R2 one-shot supervisor with explicit absolute child ledger routing.

The parent class supplies the durable request/result ledger, actual wait status,
crash/timeout handling, and no-retry directory reservation. This adapter fixes
an integration hazard: its use of setdefault otherwise allows a stale inherited
FACTORY_DURABLE_EVENT_DIR to divert child events to another experiment. No
physics or policy code is imported here.
"""
from pathlib import Path
from typing import Mapping, Sequence

from scripts.g4s2a_r2_r2_supervisor import ChildSupervisor


class R3ChildSupervisor(ChildSupervisor):
    def __init__(self, directory: str | Path, command: Sequence[str], *,
                 env: Mapping[str, str] | None = None, **kwargs):
        directory = Path(directory).absolute()
        child_env = {} if env is None else dict(env)
        child_env["FACTORY_DURABLE_EVENT_DIR"] = str(directory / "events")
        super().__init__(directory, command, env=child_env, **kwargs)
