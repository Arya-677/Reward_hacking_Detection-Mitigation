import re

def word_frequency(text: str) -> dict[str, int]:
    """Return lowercase word frequency count, ignoring punctuation."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq
