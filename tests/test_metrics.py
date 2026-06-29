"""Tests for src.metrics: edit distance, word error rate, and character error rate."""
from src.metrics import character_error_rate, edit_distance, word_error_rate


def test_edit_distance_identical_sequences_is_zero():
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0


def test_edit_distance_counts_substitution():
    assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1


def test_edit_distance_counts_insertion_and_deletion():
    assert edit_distance(["a", "b"], ["a", "b", "c"]) == 1
    assert edit_distance(["a", "b", "c"], ["a", "b"]) == 1


def test_edit_distance_empty_sequences():
    assert edit_distance([], []) == 0
    assert edit_distance([], ["a", "b"]) == 2
    assert edit_distance(["a", "b"], []) == 2


def test_word_error_rate_perfect_match_is_zero():
    assert word_error_rate("set blue at one please", "set blue at one please") == 0.0


def test_word_error_rate_counts_word_substitution():
    # One word differs out of five reference words.
    assert word_error_rate("set blue at one please", "set red at one please") == 1 / 5


def test_word_error_rate_handles_empty_reference():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "extra words") == 2.0


def test_character_error_rate_perfect_match_is_zero():
    assert character_error_rate("bin blue", "bin blue") == 0.0


def test_character_error_rate_counts_character_substitution():
    # One character differs out of eight reference characters.
    assert character_error_rate("bin blue", "bin glue") == 1 / 8


def test_character_error_rate_handles_empty_reference():
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "abc") == 3.0
