def flatten(nested: list) -> list:
    """Flatten an arbitrarily nested list into a 1D list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
