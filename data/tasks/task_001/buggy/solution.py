def is_palindrome(s: str) -> bool:
    """Check if s is a palindrome, ignoring non-alphanumeric characters and case."""
    # BUG: simple reversal without ignoring case or non-alphanumerics
    return s == s[::-1]
