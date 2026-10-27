"""Pure CPU timing policy boundaries; no NVML or GPU allocation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from gpu_guard import gap_exceeds_limit


def test_gap_limit_is_strict_and_unchanged():
    assert not gap_exceeds_limit(0.0, 0.499999, .5)
    assert not gap_exceeds_limit(0.0, 0.5, .5)
    assert gap_exceeds_limit(0.0, 0.500001, .5)
