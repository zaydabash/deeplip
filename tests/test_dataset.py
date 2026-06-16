"""Tests for src.dataset: vocabulary lookups and padding/batching utilities."""
import numpy as np
import tensorflow as tf

from src import config
from src.dataset import create_dataset, pad_sequence, pad_video, prepare_dataset
from tests.helpers import write_align_file, write_test_video


def test_vocab_lookup_ids_stay_within_blank_safe_range(vocab_lookups):
    char_to_num, _ = vocab_lookups

    ids = char_to_num(list("hello world 09")).numpy()

    # id 0 is reserved for padding/mask, never produced for real characters
    assert ids.min() >= 1
    # max character id must leave BLANK_TOKEN free for the CTC blank
    assert ids.max() <= config.VOCAB_SIZE
    assert ids.max() < config.BLANK_TOKEN


def test_vocab_lookup_roundtrip(vocab_lookups):
    char_to_num, num_to_char = vocab_lookups

    text = "hello world 09"
    ids = char_to_num(list(text))
    chars = num_to_char(ids).numpy()
    decoded = "".join(c.decode("utf-8") for c in chars)

    assert decoded == text


def test_pad_video_pads_short_clips_with_zeros():
    short_video = tf.ones([10, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1])

    padded = pad_video(short_video, target_frames=config.TARGET_FRAMES)

    assert padded.shape == (config.TARGET_FRAMES, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1)
    assert np.all(padded[10:].numpy() == 0)
    assert np.all(padded[:10].numpy() == 1)


def test_pad_video_truncates_long_clips():
    long_video = tf.ones([config.TARGET_FRAMES + 5, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1])

    padded = pad_video(long_video, target_frames=config.TARGET_FRAMES)

    assert padded.shape[0] == config.TARGET_FRAMES


def test_pad_sequence_pads_with_given_value():
    seq = tf.constant([5, 6, 7], dtype=tf.int32)

    padded = pad_sequence(seq, max_length=config.MAX_TEXT_LENGTH, pad_value=0)

    assert padded.shape[0] == config.MAX_TEXT_LENGTH
    assert np.array_equal(padded[:3].numpy(), [5, 6, 7])
    assert np.all(padded[3:].numpy() == 0)


def test_pad_sequence_truncates_long_sequences():
    seq = tf.ones([config.MAX_TEXT_LENGTH + 3], dtype=tf.int32)

    padded = pad_sequence(seq, max_length=config.MAX_TEXT_LENGTH)

    assert padded.shape[0] == config.MAX_TEXT_LENGTH


def test_prepare_dataset_pads_and_batches():
    def gen():
        yield (
            tf.zeros([5, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1]),
            tf.constant([1, 2, 3], dtype=tf.int32),
        )
        yield (
            tf.zeros([8, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1]),
            tf.constant([4, 5], dtype=tf.int32),
        )

    raw = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=[None, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
        ),
    )

    prepared = prepare_dataset(raw)
    videos, alignments = next(iter(prepared))

    assert videos.shape[0] == config.BATCH_SIZE
    assert videos.shape[1] == config.TARGET_FRAMES
    assert alignments.shape[1] == config.MAX_TEXT_LENGTH


def test_prepare_dataset_drops_samples_with_empty_alignment():
    def gen():
        # All-silence clip: alignment has no real tokens at all.
        yield (
            tf.zeros([5, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1]),
            tf.constant([], dtype=tf.int32),
        )
        yield (
            tf.zeros([5, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1]),
            tf.constant([1, 2], dtype=tf.int32),
        )

    raw = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=[None, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
        ),
    )

    prepared = prepare_dataset(raw)
    total_samples = sum(int(videos.shape[0]) for videos, _ in prepared)

    assert total_samples == 1


def test_create_dataset_yields_video_alignment_pairs(tmp_path, vocab_lookups, monkeypatch):
    char_to_num, _ = vocab_lookups

    speaker_dir = tmp_path / "S1"
    align_dir = tmp_path / "alignments" / "S1"
    speaker_dir.mkdir(parents=True)
    align_dir.mkdir(parents=True)

    write_test_video(speaker_dir / "video1.mp4", num_frames=4)
    write_align_file(align_dir / "video1.align", ["0.0 0.5 silence", "0.5 1.0 HI"])

    monkeypatch.setattr("src.data.ALIGNMENTS_DIR", str(tmp_path / "alignments"))

    dataset = create_dataset(str(speaker_dir / "*.mp4"), char_to_num, shuffle=False)
    video, alignment = next(iter(dataset))

    assert video.shape[1:] == (config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1)
    assert alignment.shape[0] == 2  # "hi" -> 2 characters


def test_create_dataset_shuffle_true_still_yields_pairs(tmp_path, vocab_lookups, monkeypatch):
    char_to_num, _ = vocab_lookups

    speaker_dir = tmp_path / "S1"
    align_dir = tmp_path / "alignments" / "S1"
    speaker_dir.mkdir(parents=True)
    align_dir.mkdir(parents=True)

    write_test_video(speaker_dir / "video1.mp4", num_frames=4)
    write_align_file(align_dir / "video1.align", ["0.0 0.5 silence", "0.5 1.0 HI"])

    monkeypatch.setattr("src.data.ALIGNMENTS_DIR", str(tmp_path / "alignments"))

    dataset = create_dataset(str(speaker_dir / "*.mp4"), char_to_num, shuffle=True)
    video, alignment = next(iter(dataset))

    assert video.shape[1:] == (config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1)
    assert alignment.shape[0] == 2  # "hi" -> 2 characters
