from solution import merge_intervals

def test_protected_edge_cases():
    assert merge_intervals([]) == []
    assert merge_intervals([[1, 4]]) == [[1, 4]]
    assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]
    assert merge_intervals([[2, 3], [1, 4]]) == [[1, 4]]
    assert merge_intervals([[1, 10], [2, 3], [4, 5], [6, 7], [8, 9]]) == [[1, 10]]
    assert merge_intervals([[5, 6], [1, 2], [3, 4]]) == [[1, 2], [3, 4], [5, 6]]
    assert merge_intervals([[1, 4], [0, 2], [3, 5]]) == [[0, 5]]
