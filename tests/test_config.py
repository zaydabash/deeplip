"""Tests for src.config invariants relied on by the model/dataset/loss pipeline."""
from src import config


def test_vocab_size_matches_vocab_length():
    assert config.VOCAB_SIZE == len(config.VOCAB)


def test_blank_token_does_not_collide_with_character_ids():
    # Character ids occupy 1..VOCAB_SIZE (StringLookup with num_oov_indices=0,
    # id 0 reserved for padding/mask). BLANK_TOKEN must sit one past the last
    # character id so the CTC blank never collides with a real character.
    assert config.BLANK_TOKEN == config.VOCAB_SIZE + 1
    assert config.BLANK_TOKEN > config.VOCAB_SIZE


def test_video_dimensions_match_mouth_region():
    region = config.MOUTH_REGION

    assert config.VIDEO_HEIGHT == region["bottom"] - region["top"]
    assert config.VIDEO_WIDTH == region["right"] - region["left"]
    assert config.VIDEO_HEIGHT > 0
    assert config.VIDEO_WIDTH > 0
