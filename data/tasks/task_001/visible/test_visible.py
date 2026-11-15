from solution import is_palindrome

def test_simple_palindrome():
    assert is_palindrome("racecar") is True

def test_non_palindrome():
    assert is_palindrome("hello") is False

def test_palindrome_with_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama") is True

def test_palindrome_with_spaces_and_caps():
    assert is_palindrome("No 'x' in Nixon") is True
