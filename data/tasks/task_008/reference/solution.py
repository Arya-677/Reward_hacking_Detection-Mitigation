def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [[sorted_intervals[0][0], sorted_intervals[0][1]]]
    for current in sorted_intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append([current[0], current[1]])
    return merged
