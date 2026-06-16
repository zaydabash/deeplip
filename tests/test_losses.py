"""Tests for src.losses.ctc_loss_fn."""
import numpy as np
import tensorflow as tf

from src import config
from src.losses import ctc_loss_fn


def _random_predictions(batch_size: int) -> tf.Tensor:
    logits = tf.random.uniform([batch_size, config.TARGET_FRAMES, config.BLANK_TOKEN + 1])
    return tf.nn.softmax(logits, axis=-1)


def test_ctc_loss_is_finite_for_valid_labels():
    # Includes VOCAB_SIZE (the highest character id, "space"), which used to
    # collide with the CTC blank index before the BLANK_TOKEN fix.
    y_true = tf.constant(
        [
            [1, 2, config.VOCAB_SIZE, 0, 0],
            [3, 4, 5, 0, 0],
        ],
        dtype=tf.int32,
    )

    loss = ctc_loss_fn(y_true, _random_predictions(batch_size=2))

    assert np.isfinite(loss.numpy())
    assert loss.numpy() > 0


def test_ctc_loss_handles_minimal_single_token_label():
    # A clip whose alignment has exactly one real character (the smallest
    # valid label tf.nn.ctc_loss accepts; an all-padding row would crash it).
    y_true = tf.constant([[1, 0, 0, 0, 0]], dtype=tf.int32)

    loss = ctc_loss_fn(y_true, _random_predictions(batch_size=1))

    assert np.isfinite(loss.numpy())
