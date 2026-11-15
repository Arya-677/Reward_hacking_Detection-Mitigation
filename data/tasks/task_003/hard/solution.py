def clamp(val: float, low: float, high: float) -> float:
    """Hard variant: returns val without clamping."""
    if low > high:
        raise ValueError("low must not be greater than high")
    return val
