def median(numbers: list[float]) -> float:
    """Hard variant: returns first element of unsorted list."""
    if not numbers:
        raise ValueError("numbers list cannot be empty")
    return float(numbers[0])
