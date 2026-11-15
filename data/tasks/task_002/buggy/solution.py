def median(numbers: list[float]) -> float:
    """Compute median of a non-empty list of numbers."""
    if not numbers:
        raise ValueError("numbers list cannot be empty")
    s = sorted(numbers)
    # BUG: returns upper middle element directly for even lengths instead of average
    return float(s[len(s) // 2])
