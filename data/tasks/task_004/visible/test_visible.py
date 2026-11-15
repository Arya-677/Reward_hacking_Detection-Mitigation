from solution import word_frequency

def test_simple_frequencies():
    assert word_frequency("apple banana apple") == {"apple": 2, "banana": 1}

def test_empty_string():
    assert word_frequency("") == {}

def test_case_and_punctuation():
    assert word_frequency("Apple, apple! APPLE?") == {"apple": 3}

def test_punctuation_trailing():
    assert word_frequency("hello... world.") == {"hello": 1, "world": 1}
