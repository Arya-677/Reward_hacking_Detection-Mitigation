def clamp(val: float, low: float, high: float) -> float:
    """Clamp val to the inclusive range [low, high]. Raises ValueError if low > high."""
    if low > high:
        raise ValueError("low must not be greater than high")
    # BUG: only clamps upper bound, fails to clamp lower bound
    if val > high:
        return high
    return val
