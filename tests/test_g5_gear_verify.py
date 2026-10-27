from scripts.g5_gear_verify import geometric

def test_strict_thresholds_and_one_sided_height():
    origin = [0.,0.,0.]
    assert geometric([.00249, 0, .00099], origin)
    assert not geometric([.0025, 0, 0], origin)
    assert not geometric([0, 0, .001], origin)
    assert geometric([0, 0, -.1], origin)
    assert not geometric([.002, .002, 0], origin)
