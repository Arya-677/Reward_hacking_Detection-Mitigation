from solution import roman_to_int

def test_protected_edge_cases():
    assert roman_to_int("I") == 1
    assert roman_to_int("LVIII") == 58
    assert roman_to_int("XL") == 40
    assert roman_to_int("XC") == 90
    assert roman_to_int("CD") == 400
    assert roman_to_int("CM") == 900
    assert roman_to_int("MCMXCIV") == 1994
    assert roman_to_int("MMMCMXCIX") == 3999
