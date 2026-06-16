"""Tests for src.predict: numpy padding, CTC decoding, and inference helpers."""
import numpy as np

from src import config
from src.predict import decode_predictions, load_model, main, pad_video, predict_clip
from tests.helpers import write_test_video


def test_pad_video_pads_short_clip_with_zeros():
    video = np.ones((10, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1), dtype=np.float32)

    padded = pad_video(video, target_frames=config.TARGET_FRAMES)

    assert padded.shape == (config.TARGET_FRAMES, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1)
    assert np.all(padded[10:] == 0)
    assert np.all(padded[:10] == 1)


def test_pad_video_truncates_long_clip():
    video = np.ones((config.TARGET_FRAMES + 5, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1), dtype=np.float32)

    padded = pad_video(video, target_frames=config.TARGET_FRAMES)

    assert padded.shape[0] == config.TARGET_FRAMES


def _one_hot_predictions(class_sequence, num_classes):
    timesteps = len(class_sequence)
    predictions = np.zeros((1, timesteps, num_classes), dtype=np.float32)
    for t, class_id in enumerate(class_sequence):
        predictions[0, t, class_id] = 1.0
    return predictions


def test_decode_predictions_decodes_greedy_path(vocab_lookups):
    char_to_num, num_to_char = vocab_lookups
    num_classes = config.BLANK_TOKEN + 1

    text = "hi"
    char_ids = char_to_num(list(text)).numpy()

    # Greedy decode collapses repeats and drops the CTC blank (the last class).
    sequence = [char_ids[0], config.BLANK_TOKEN, char_ids[1], config.BLANK_TOKEN]
    predictions = _one_hot_predictions(sequence, num_classes)

    assert decode_predictions(predictions, num_to_char) == text


def test_decode_predictions_returns_empty_string_for_all_blank(vocab_lookups):
    _, num_to_char = vocab_lookups
    num_classes = config.BLANK_TOKEN + 1

    sequence = [config.BLANK_TOKEN] * 4
    predictions = _one_hot_predictions(sequence, num_classes)

    assert decode_predictions(predictions, num_to_char) == ""


def test_predict_clip_returns_decoded_string(tmp_path, vocab_lookups, model):
    _, num_to_char = vocab_lookups

    video_path = tmp_path / "clip.mp4"
    write_test_video(video_path, num_frames=4)

    text = predict_clip(str(video_path), model, num_to_char)

    assert isinstance(text, str)


def test_load_model_with_missing_weights_uses_random_init(tmp_path, capsys):
    load_model(weights_path=str(tmp_path / "missing.h5"))

    captured = capsys.readouterr()
    assert "Warning: Weights file not found" in captured.out


def test_load_model_loads_existing_weights(tmp_path, model, capsys):
    weights_path = tmp_path / "weights.h5"
    model.save_weights(str(weights_path))

    load_model(weights_path=str(weights_path))

    captured = capsys.readouterr()
    assert "Weights loaded successfully" in captured.out


def test_main_runs_end_to_end_prediction(tmp_path, capsys):
    video_path = tmp_path / "clip.mp4"
    write_test_video(video_path, num_frames=4)

    main(video_path=str(video_path), weights_path=str(tmp_path / "missing.h5"))

    captured = capsys.readouterr()
    assert "PREDICTION RESULT" in captured.out
    assert "Predicted text:" in captured.out
