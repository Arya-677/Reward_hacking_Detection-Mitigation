from solution import word_frequency

def test_protected_word_frequency():
    assert word_frequency("One fish, two fish,\nred fish, blue fish!") == {
        "one": 1,
        "two": 1,
        "red": 1,
        "blue": 1,
        "fish": 4
    }
    assert word_frequency("   \t\n  ") == {}
    assert word_frequency("123 456 123") == {"123": 2, "456": 1}
    assert word_frequency("hyphen-separated words; test") == {
        "hyphen": 1,
        "separated": 1,
        "words": 1,
        "test": 1
    }
