"""Vocabulary lookups and the tf.data input pipeline."""
import glob

import tensorflow as tf

from . import config
from .data import load_data_tf


def build_vocab_lookup():
    """Build (char_to_num, num_to_char) StringLookup layers.

    Characters map to ids 1..VOCAB_SIZE; id 0 is the reserved OOV/padding index
    so the CTC blank (BLANK_TOKEN) never collides with a real character.
    """
    char_to_num = tf.keras.layers.StringLookup(
        vocabulary=list(config.VOCAB), oov_token="", mask_token=None
    )
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        oov_token="",
        mask_token=None,
        invert=True,
    )
    return char_to_num, num_to_char


def pad_video(video, target_frames=config.TARGET_FRAMES):
    """Pad with zero frames (or truncate) to exactly target_frames."""
    video = video[:target_frames]
    pad = tf.maximum(target_frames - tf.shape(video)[0], 0)
    paddings = [[0, pad], [0, 0], [0, 0], [0, 0]]
    padded = tf.pad(video, paddings)
    padded.set_shape([target_frames, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1])
    return padded


def pad_sequence(seq, max_length=config.MAX_TEXT_LENGTH, pad_value=0):
    """Pad (or truncate) a 1-D sequence to exactly max_length."""
    seq = seq[:max_length]
    pad = tf.maximum(max_length - tf.shape(seq)[0], 0)
    padded = tf.pad(seq, [[0, pad]], constant_values=pad_value)
    padded.set_shape([max_length])
    return padded


def prepare_dataset(dataset):
    """Drop empty-alignment samples, pad videos/labels, and batch."""
    dataset = dataset.filter(lambda video, alignment: tf.shape(alignment)[0] > 0)
    dataset = dataset.map(
        lambda video, alignment: (pad_video(video), pad_sequence(tf.cast(alignment, tf.int32))),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(config.BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def create_dataset(video_pattern, char_to_num, shuffle=True):
    """Build an unbatched dataset of (video, alignment) pairs from a glob pattern.

    When shuffle is True the order is fixed for the lifetime of the dataset
    (reshuffle_each_iteration=False) so that take()/skip() splits stay disjoint.
    """
    files = sorted(glob.glob(video_pattern))
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(files), 1), reshuffle_each_iteration=False)

    def _load(path):
        video, alignment = tf.py_function(
            func=lambda p: load_data_tf(p, char_to_num),
            inp=[path],
            Tout=(tf.float32, tf.int32),
        )
        video.set_shape([None, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1])
        alignment.set_shape([None])
        return video, alignment

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
