from solution import median

def test_odd_length_list():
    assert median([3, 1, 2]) == 2.0

def test_single_element():
    assert median([5]) == 5.0

def test_even_length_list():
    assert median([1, 2, 3, 4]) == 2.5

def test_even_length_larger_numbers():
    assert median([10, 20, 30, 40]) == 25.0
