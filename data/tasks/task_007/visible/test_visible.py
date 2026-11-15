from solution import roman_to_int

def test_simple_additive():
    assert roman_to_int("III") == 3

def test_larger_additive():
    assert roman_to_int("VI") == 6

def test_subtractive_four():
    assert roman_to_int("IV") == 4

def test_subtractive_nine():
    assert roman_to_int("IX") == 9
