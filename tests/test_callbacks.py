"""Tests for src.callbacks: learning rate schedule, checkpointing, and example output."""
import numpy as np
import tensorflow as tf

from src import config
from src.callbacks import LearningRateSchedule, ModelCheckpoint, ProduceExample
from src.config import INITIAL_LEARNING_RATE, LR_DECAY_RATE, LR_DECAY_START_EPOCH


def _compiled(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE), loss="mse")
    return model


def test_learning_rate_is_constant_before_decay_start(model):
    _compiled(model)
    callback = LearningRateSchedule()
    callback.model = model

    callback.on_epoch_begin(epoch=0)

    lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    assert np.isclose(lr, INITIAL_LEARNING_RATE)


def test_learning_rate_decays_starting_at_threshold(model):
    _compiled(model)
    callback = LearningRateSchedule()
    callback.model = model

    callback.on_epoch_begin(epoch=LR_DECAY_START_EPOCH)

    expected = INITIAL_LEARNING_RATE * LR_DECAY_RATE
    lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    assert np.isclose(lr, expected, rtol=1e-5)


def test_model_checkpoint_saves_weights_file(tmp_path, model):
    checkpoint = ModelCheckpoint(save_dir=str(tmp_path))
    checkpoint.model = model

    checkpoint.on_epoch_end(epoch=0)

    assert (tmp_path / "weights_epoch_01.h5").exists()


class _AllBlankModel:
    """Stand-in model whose predictions decode to the empty string for every example."""

    def predict(self, videos, verbose=0):
        batch_size = videos.shape[0]
        predictions = np.zeros((batch_size, config.TARGET_FRAMES, config.BLANK_TOKEN + 1), dtype=np.float32)
        predictions[:, :, config.BLANK_TOKEN] = 1.0
        return predictions


def test_produce_example_handles_empty_ground_truth_and_prediction(vocab_lookups, capsys):
    _, num_to_char = vocab_lookups

    videos = tf.zeros([2, config.TARGET_FRAMES, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1])
    # Second example has no real label at all (all padding).
    labels = tf.constant(
        [[1, 2] + [0] * (config.MAX_TEXT_LENGTH - 2), [0] * config.MAX_TEXT_LENGTH],
        dtype=tf.int32,
    )
    val_dataset = tf.data.Dataset.from_tensors((videos, labels))

    callback = ProduceExample(val_dataset=val_dataset, num_to_char=num_to_char)
    callback.model = _AllBlankModel()

    callback.on_epoch_end(epoch=0)

    captured = capsys.readouterr()
    # All-blank predictions decode to "" for both examples, and the
    # all-padding second label also has empty ground truth.
    assert "Ground Truth: \n" in captured.out
    assert "Predicted:    \n" in captured.out


def test_produce_example_prints_ground_truth_and_prediction(model, vocab_lookups, capsys):
    _, num_to_char = vocab_lookups

    videos = tf.zeros([2, config.TARGET_FRAMES, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1])
    labels = tf.constant(
        [[1, 2] + [0] * (config.MAX_TEXT_LENGTH - 2), [3] + [0] * (config.MAX_TEXT_LENGTH - 1)],
        dtype=tf.int32,
    )
    val_dataset = tf.data.Dataset.from_tensors((videos, labels))

    callback = ProduceExample(val_dataset=val_dataset, num_to_char=num_to_char)
    callback.model = model

    callback.on_epoch_end(epoch=0)

    captured = capsys.readouterr()
    assert "Epoch 1 - Example Predictions" in captured.out
    assert "Ground Truth" in captured.out
    assert "Predicted" in captured.out


class _PartialDecodeModel:
    """Predicts one real character for example 0, and blanks everywhere else."""

    def predict(self, videos, verbose=0):
        batch_size = videos.shape[0]
        predictions = np.zeros((batch_size, config.TARGET_FRAMES, config.BLANK_TOKEN + 1), dtype=np.float32)
        predictions[:, :, config.BLANK_TOKEN] = 1.0
        # Example 0, first timestep: predict character id 1 ("a") instead of blank.
        predictions[0, 0, config.BLANK_TOKEN] = 0.0
        predictions[0, 0, 1] = 1.0
        return predictions


def test_produce_example_prints_non_empty_prediction_text(vocab_lookups, capsys):
    _, num_to_char = vocab_lookups

    videos = tf.zeros([2, config.TARGET_FRAMES, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1])
    labels = tf.constant(
        [[1, 2] + [0] * (config.MAX_TEXT_LENGTH - 2), [0] * config.MAX_TEXT_LENGTH],
        dtype=tf.int32,
    )
    val_dataset = tf.data.Dataset.from_tensors((videos, labels))

    callback = ProduceExample(val_dataset=val_dataset, num_to_char=num_to_char)
    callback.model = _PartialDecodeModel()

    callback.on_epoch_end(epoch=0)

    captured = capsys.readouterr()
    # Example 0's lone non-blank prediction (character id 1) decodes to "a".
    assert "Predicted:    a" in captured.out
