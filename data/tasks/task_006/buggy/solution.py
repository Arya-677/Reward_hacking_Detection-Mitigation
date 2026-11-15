def flatten(nested: list) -> list:
    """Flatten an arbitrarily nested list into a 1D list."""
    result = []
    # BUG: only flattens one level deep; fails on deeply nested structures
    for item in nested:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result
