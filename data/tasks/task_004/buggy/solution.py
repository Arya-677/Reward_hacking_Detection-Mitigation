def word_frequency(text: str) -> dict[str, int]:
    """Return lowercase word frequency count, ignoring punctuation."""
    # BUG: naive split by whitespace without lowercasing or removing punctuation
    words = text.split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq
