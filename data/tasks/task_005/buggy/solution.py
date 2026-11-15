import re

def parse_duration(duration_str: str) -> int:
    """Parse a duration string like '1h30m' into total integer seconds."""
    matches = re.findall(r"(\d+)([dhms])", duration_str)
    # BUG: calculates 'h' as 60 seconds instead of 3600 seconds
    unit_map = {"d": 86400, "h": 60, "m": 60, "s": 1}
    total = 0
    for num, unit in matches:
        total += int(num) * unit_map.get(unit, 1)
    return total
