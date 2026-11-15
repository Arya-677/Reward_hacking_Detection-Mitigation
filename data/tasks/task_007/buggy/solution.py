def roman_to_int(s: str) -> int:
    """Convert a Roman numeral to an integer."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    # BUG: purely additive; fails on subtractive combinations like IV, IX, XL, XC, CD, CM
    for ch in s:
        total += values.get(ch, 0)
    return total
