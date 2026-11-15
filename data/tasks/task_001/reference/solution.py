def is_palindrome(s: str) -> bool:
    """Check if s is a palindrome, ignoring non-alphanumeric characters and case."""
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
