"""CPU-only, atomic evaluation evidence and binomial interval helpers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def atomic_write_json(path: Path, result: dict) -> None:
    """Replace one progress snapshot atomically, preserving valid prior JSON.

    A killed writer can leave ``.tmp``; readers use only the final path. This
    helper rejects NaN/Infinity, and rates always retain their actual denominator.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    """Two-sided 95% Wilson score interval; null for no completed episodes.

    The episode-binomial interval is descriptive: vector environments share a
    policy and seeds, so it does not quantify training-seed uncertainty.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("require 0 <= successes <= denominator")
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    # Preserve the mathematical endpoints despite float roundoff.
    return [0.0 if successes == 0 else max(0.0, center - half_width),
            1.0 if successes == n else min(1.0, center + half_width)]


def add_intervals(aggregate: dict) -> dict:
    result = dict(aggregate)
    n = aggregate["episodes_completed"]
    for name in ("ever_success", "terminal_success"):
        result[name + "_wilson95"] = wilson_interval(aggregate[name + "_count"], n)
    return result
