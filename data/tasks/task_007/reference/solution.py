def roman_to_int(s: str) -> int:
    """Convert a Roman numeral to an integer."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_val = 0
    for ch in reversed(s):
        curr_val = values.get(ch, 0)
        if curr_val < prev_val:
            total -= curr_val
        else:
            total += curr_val
            prev_val = curr_val
    return total
