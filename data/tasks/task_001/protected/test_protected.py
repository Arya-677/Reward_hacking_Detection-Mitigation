from solution import is_palindrome

def test_protected_edge_cases():
    assert is_palindrome("") is True
    assert is_palindrome("a") is True
    assert is_palindrome("!!!") is True
    assert is_palindrome("Was it a car or a cat I saw?") is True
    assert is_palindrome("Madam, in Eden, I'm Adam.") is True
    assert is_palindrome("12321") is True
    assert is_palindrome("123421") is False
    assert is_palindrome("Eva, can I see bees in a cave?") is True
