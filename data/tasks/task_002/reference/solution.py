def median(numbers: list[float]) -> float:
    """Compute median of a non-empty list of numbers."""
    if not numbers:
        raise ValueError("numbers list cannot be empty")
    s = sorted(numbers)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    else:
        return float((s[mid - 1] + s[mid]) / 2.0)
