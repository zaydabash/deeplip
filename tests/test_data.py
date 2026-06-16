"""Tests for src.data: video loading and alignment parsing.

Uses small synthetic fixtures (generated on the fly) rather than the real
GRID/MIRACL-VC1 clips in data/, since data/ is gitignored and not available in CI.
"""
import numpy as np
import tensorflow as tf

from src import config
from src.data import load_alignments, load_data, load_data_tf, load_video
from tests.helpers import write_align_file as _write_align_file
from tests.helpers import write_test_video as _write_test_video


def test_load_video_returns_normalized_mouth_region(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_test_video(video_path, num_frames=5)

    video = load_video(str(video_path))

    assert video.shape[0] == 5
    assert video.shape[1:] == (config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1)
    assert video.dtype == np.float32
    # Standardized (zero mean, unit variance) since the frames have nonzero std
    assert abs(float(video.mean())) < 1e-3


def test_load_video_handles_constant_frames_without_division_by_zero(tmp_path):
    video_path = tmp_path / "constant.mp4"
    # All frames the same value -> zero standard deviation, exercising the
    # mean-subtraction-only branch instead of (x - mean) / std.
    _write_test_video(video_path, num_frames=3, frame_value=50)

    video = load_video(str(video_path))

    assert video.shape[0] == 3
    assert np.all(video == 0.0)


def test_load_video_raises_for_empty_video(tmp_path):
    video_path = tmp_path / "empty.mp4"
    _write_test_video(video_path, num_frames=0)

    try:
        load_video(str(video_path))
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_load_alignments_filters_silence_and_lowercases(tmp_path, vocab_lookups):
    char_to_num, num_to_char = vocab_lookups

    align_path = tmp_path / "clip.align"
    _write_align_file(
        align_path,
        ["0.0 0.5 SILENCE", "0.5 1.0 HI", "1.0 1.5 silence"],
    )

    ids = load_alignments(str(align_path), char_to_num)
    chars = num_to_char(ids).numpy()
    text = "".join(c.decode("utf-8") for c in chars)

    assert text == "hi"


def test_load_alignments_empty_when_only_silence(tmp_path, vocab_lookups):
    char_to_num, _ = vocab_lookups

    align_path = tmp_path / "silent.align"
    _write_align_file(align_path, ["0.0 1.0 silence"])

    ids = load_alignments(str(align_path), char_to_num)

    assert len(ids) == 0


def test_load_data_pairs_video_with_its_alignment(tmp_path, vocab_lookups, monkeypatch):
    char_to_num, _ = vocab_lookups

    speaker_dir = tmp_path / "S1"
    align_dir = tmp_path / "alignments" / "S1"
    speaker_dir.mkdir(parents=True)
    align_dir.mkdir(parents=True)

    video_path = speaker_dir / "video1.mp4"
    _write_test_video(video_path, num_frames=4)
    _write_align_file(align_dir / "video1.align", ["0.0 0.5 silence", "0.5 1.0 HI"])

    monkeypatch.setattr("src.data.ALIGNMENTS_DIR", str(tmp_path / "alignments"))

    video, alignment = load_data(str(video_path), char_to_num)

    assert video.shape[0] == 4
    assert len(alignment) == 2  # "hi" -> 2 characters


def test_load_data_tf_wraps_load_data_for_tf_data_pipeline(tmp_path, vocab_lookups, monkeypatch):
    char_to_num, _ = vocab_lookups

    speaker_dir = tmp_path / "S1"
    align_dir = tmp_path / "alignments" / "S1"
    speaker_dir.mkdir(parents=True)
    align_dir.mkdir(parents=True)

    video_path = speaker_dir / "video1.mp4"
    _write_test_video(video_path, num_frames=4)
    _write_align_file(align_dir / "video1.align", ["0.0 0.5 silence", "0.5 1.0 HI"])

    monkeypatch.setattr("src.data.ALIGNMENTS_DIR", str(tmp_path / "alignments"))

    video, alignment = load_data_tf(tf.constant(str(video_path)), char_to_num)

    assert video.shape[0] == 4
    assert video.shape[1:] == (config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1)
    assert alignment.shape[0] == 2  # "hi" -> 2 characters
