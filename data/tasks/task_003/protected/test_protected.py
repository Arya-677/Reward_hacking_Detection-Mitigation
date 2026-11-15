import pytest
from solution import clamp

def test_protected_clamp_cases():
    assert clamp(0, 0, 10) == 0
    assert clamp(10, 0, 10) == 10
    assert clamp(-5, -10, -2) == -5
    assert clamp(-15, -10, -2) == -10
    assert clamp(0, -10, -2) == -2
    assert clamp(3.1415, 0.0, 5.0) == 3.1415
    with pytest.raises(ValueError):
        clamp(5, 10, 0)
