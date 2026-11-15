def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    # BUG: assumes intervals are already sorted by start time
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)
    return merged
