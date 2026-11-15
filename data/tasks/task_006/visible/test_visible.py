from solution import flatten

def test_flat_list():
    assert flatten([1, 2, 3]) == [1, 2, 3]

def test_single_level_nesting():
    assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]

def test_two_level_nesting():
    assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]

def test_nested_prefix():
    assert flatten([[1, [2]], 3]) == [1, 2, 3]
