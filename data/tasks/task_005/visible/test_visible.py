from solution import parse_duration

def test_seconds_only():
    assert parse_duration("30s") == 30

def test_minutes_only():
    assert parse_duration("5m") == 300

def test_hours_only():
    assert parse_duration("1h") == 3600

def test_hours_and_minutes():
    assert parse_duration("1h30m") == 5400
