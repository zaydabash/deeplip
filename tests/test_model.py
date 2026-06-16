"""Tests for src.model: architecture shapes and parameter counts."""
from src import config
from src.model import print_model_summary


def test_model_input_shape(model):
    assert model.input_shape == (None, config.TARGET_FRAMES, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1)


def test_model_output_shape_includes_blank_token(model):
    # Output units = character ids (1..VOCAB_SIZE) + dedicated blank token (BLANK_TOKEN)
    # + the reserved padding id 0 => VOCAB_SIZE + 2 == BLANK_TOKEN + 1
    assert model.output_shape == (None, config.TARGET_FRAMES, config.BLANK_TOKEN + 1)


def test_model_has_trainable_parameters(model):
    assert model.count_params() > 0
    assert all(layer.count_params() >= 0 for layer in model.layers)


def test_print_model_summary_reports_total_params(model, capsys):
    print_model_summary(model)

    captured = capsys.readouterr()
    assert "Total parameters" in captured.out
    assert f"{model.count_params():,}" in captured.out
