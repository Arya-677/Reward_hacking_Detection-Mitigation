def clamp(val: float, low: float, high: float) -> float:
    """Clamp val to the inclusive range [low, high]. Raises ValueError if low > high."""
    if low > high:
        raise ValueError("low must not be greater than high")
    if val < low:
        return low
    if val > high:
        return high
    return val
