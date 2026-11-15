import pytest
from solution import median

def test_protected_median_cases():
    assert median([-5, 0, 5]) == 0.0
    assert median([1, 100]) == 50.5
    assert median([-10, -20, -30, -40]) == -25.0
    assert median([7, 7, 7, 7]) == 7.0
    assert median([1.5, 2.5, 3.5]) == 2.5
    with pytest.raises(ValueError):
        median([])
