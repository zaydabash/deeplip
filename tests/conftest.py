"""Shared pytest fixtures for the test suite."""
import pytest

from src.dataset import build_vocab_lookup
from src.model import build_model


@pytest.fixture(scope="session")
def vocab_lookups():
    """Char-to-num / num-to-char StringLookup layers, built once per session."""
    return build_vocab_lookup()


@pytest.fixture(scope="session")
def model():
    """A freshly built (untrained) lip-reading model, built once per session."""
    return build_model()
