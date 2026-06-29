"""Word/character error rate metrics for scoring predictions against ground truth."""


def edit_distance(reference, hypothesis):
    """Levenshtein distance between two sequences (e.g. lists of words or chars)."""
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    distances = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(ref_len + 1):
        distances[i][0] = i
    for j in range(hyp_len + 1):
        distances[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distances[i][j] = distances[i - 1][j - 1]
            else:
                distances[i][j] = 1 + min(
                    distances[i - 1][j],      # deletion
                    distances[i][j - 1],      # insertion
                    distances[i - 1][j - 1],  # substitution
                )

    return distances[ref_len][hyp_len]


def word_error_rate(reference, hypothesis):
    """Word-level edit distance divided by the number of words in the reference.

    If the reference has no words, WER is 0.0 when the hypothesis is also
    empty, otherwise the raw edit distance (there's no reference length to
    normalize against).
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 0.0 if not hyp_words else float(len(hyp_words))

    return edit_distance(ref_words, hyp_words) / len(ref_words)


def character_error_rate(reference, hypothesis):
    """Character-level edit distance divided by the number of characters in the reference.

    If the reference is empty, CER is 0.0 when the hypothesis is also empty,
    otherwise the raw edit distance.
    """
    if not reference:
        return 0.0 if not hypothesis else float(len(hypothesis))

    return edit_distance(reference, hypothesis) / len(reference)
