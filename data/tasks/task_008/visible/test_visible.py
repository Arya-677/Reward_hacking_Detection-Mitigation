from solution import merge_intervals

def test_sorted_overlapping():
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]

def test_touching_boundary():
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]

def test_unsorted_intervals():
    assert merge_intervals([[2, 6], [1, 3]]) == [[1, 6]]

def test_unsorted_spanning():
    assert merge_intervals([[1, 4], [0, 4]]) == [[0, 4]]
