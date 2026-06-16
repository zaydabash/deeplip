"""Training callbacks: learning-rate schedule, checkpointing, example output."""
import os

import numpy as np
import tensorflow as tf

from .config import (
    INITIAL_LEARNING_RATE,
    LR_DECAY_RATE,
    LR_DECAY_START_EPOCH,
)
from .predict import decode_predictions


class LearningRateSchedule(tf.keras.callbacks.Callback):
    """Constant LR until LR_DECAY_START_EPOCH, then exponential decay."""

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < LR_DECAY_START_EPOCH:
            lr = INITIAL_LEARNING_RATE
        else:
            lr = INITIAL_LEARNING_RATE * (LR_DECAY_RATE ** (epoch - LR_DECAY_START_EPOCH + 1))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """Save model weights to ``<save_dir>/weights_epoch_NN.h5`` after each epoch."""

    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"weights_epoch_{epoch + 1:02d}.h5")
        self.model.save_weights(path)


def _decode_label(label, num_to_char):
    """Decode a padded int label row into text, dropping the padding id 0."""
    ids = [int(x) for x in np.asarray(label) if int(x) != 0]
    if not ids:
        return ""
    chars = num_to_char(tf.constant(ids, dtype=tf.int64)).numpy()
    return "".join(c.decode("utf-8") for c in chars)


class ProduceExample(tf.keras.callbacks.Callback):
    """Print ground-truth vs predicted text for a validation batch each epoch."""

    def __init__(self, val_dataset, num_to_char):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_to_char = num_to_char

    def on_epoch_end(self, epoch, logs=None):
        videos, labels = next(iter(self.val_dataset))
        predictions = self.model.predict(videos, verbose=0)

        print(f"\nEpoch {epoch + 1} - Example Predictions")
        print("=" * 40)
        for i in range(len(labels)):
            ground_truth = _decode_label(labels[i], self.num_to_char)
            predicted = decode_predictions(predictions[i:i + 1], self.num_to_char)
            print(f"Ground Truth: {ground_truth}")
            print(f"Predicted:    {predicted}")
