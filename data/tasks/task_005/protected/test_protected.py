from solution import parse_duration

def test_protected_durations():
    assert parse_duration("1d") == 86400
    assert parse_duration("2d4h") == 187200
    assert parse_duration("10m45s") == 645
    assert parse_duration("1d2h3m4s") == 86400 + 7200 + 180 + 4
