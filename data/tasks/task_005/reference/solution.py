import re

def parse_duration(duration_str: str) -> int:
    """Parse a duration string like '1h30m' into total integer seconds."""
    s = duration_str.strip()
    if not s:
        raise ValueError("duration string cannot be empty")
    matches = re.findall(r"(\d+)([dhms])", s)
    if not matches:
        raise ValueError(f"Invalid duration format: {duration_str}")
    reconstructed = "".join(f"{num}{unit}" for num, unit in matches)
    if reconstructed != s.replace(" ", ""):
        raise ValueError(f"Unrecognized components in duration: {duration_str}")
    unit_map = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    total = 0
    for num, unit in matches:
        total += int(num) * unit_map[unit]
    return total
