from solution import clamp

def test_clamp_in_range():
    assert clamp(5, 0, 10) == 5

def test_clamp_above_high():
    assert clamp(15, 0, 10) == 10

def test_clamp_below_low():
    assert clamp(-5, 0, 10) == 0

def test_clamp_fractional_below_low():
    assert clamp(-0.1, 0.0, 1.0) == 0.0
