"""Inference: load a checkpoint and decode a video clip to text."""
import argparse
import os

import numpy as np
import tensorflow as tf

from . import config
from .data import load_video
from .dataset import build_vocab_lookup
from .model import build_model


def pad_video(video, target_frames=config.TARGET_FRAMES):
    """Numpy pad/truncate a [frames, H, W, 1] clip to exactly target_frames."""
    if video.shape[0] >= target_frames:
        return video[:target_frames]
    pad = np.zeros(
        (target_frames - video.shape[0],) + video.shape[1:], dtype=video.dtype
    )
    return np.concatenate([video, pad], axis=0)


def _greedy_decode_ids(pred_2d):
    """Greedy CTC decode a [time, classes] array into a list of character ids."""
    best = np.argmax(pred_2d, axis=-1)
    collapsed = []
    prev = -1
    for token in best:
        if token != prev:
            collapsed.append(int(token))
        prev = token
    return [t for t in collapsed if t != config.BLANK_TOKEN and t != 0]


def _beam_decode_ids(predictions, beam_width):
    """Beam-search CTC decode a [1, time, classes] array into character ids."""
    input_length = np.full(predictions.shape[0], predictions.shape[1])
    decoded, _ = tf.keras.backend.ctc_decode(
        predictions, input_length, greedy=False, beam_width=beam_width
    )
    ids = decoded[0].numpy()[0]
    return [int(t) for t in ids if t not in (-1, 0)]


def decode_predictions(predictions, num_to_char, greedy=True, beam_width=config.BEAM_WIDTH):
    """Decode model predictions ([1, time, classes] or [time, classes]) to text.

    greedy=True takes the best class at each timestep (fast, can be brittle on
    short/ambiguous tokens). greedy=False runs CTC beam search instead, which
    explores multiple candidate paths and tends to do better on those cases.
    """
    pred = np.asarray(predictions)
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]

    if greedy:
        ids = _greedy_decode_ids(pred[0])
    else:
        ids = _beam_decode_ids(pred, beam_width)

    if not ids:
        return ""
    chars = num_to_char(tf.constant(ids, dtype=tf.int64)).numpy()
    return "".join(c.decode("utf-8") for c in chars)


def load_model(weights_path=config.DEFAULT_WEIGHTS):
    """Build the model and load weights if present, otherwise random init."""
    model = build_model()
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Weights loaded successfully")
    else:
        print(
            f"Warning: Weights file not found at {weights_path}. "
            "Using random initialization."
        )
    return model


def predict_clip(video_path, model, num_to_char, greedy=True, beam_width=config.BEAM_WIDTH):
    """Preprocess a video, run the model, and return the decoded text."""
    video = load_video(video_path)
    video = pad_video(video)
    video = np.expand_dims(video, axis=0)
    predictions = model.predict(video, verbose=0)
    return decode_predictions(predictions, num_to_char, greedy=greedy, beam_width=beam_width)


def main(video_path, weights_path=config.DEFAULT_WEIGHTS, greedy=True, beam_width=config.BEAM_WIDTH):
    """Run end-to-end prediction for a single video and print the result."""
    _, num_to_char = build_vocab_lookup()
    model = load_model(weights_path)
    text = predict_clip(video_path, model, num_to_char, greedy=greedy, beam_width=beam_width)

    print("=" * 40)
    print("PREDICTION RESULT")
    print("=" * 40)
    print(f"Predicted text: {text}")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict text from a lip-reading video clip.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument(
        "--weights",
        default=config.DEFAULT_WEIGHTS,
        help="Path to model weights (.h5).",
    )
    parser.add_argument(
        "--decoding",
        choices=["greedy", "beam"],
        default="greedy",
        help="CTC decoding strategy (default: greedy).",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=config.BEAM_WIDTH,
        help=f"Beam width for beam-search decoding, ignored for greedy (default: {config.BEAM_WIDTH}).",
    )
    args = parser.parse_args()
    main(
        video_path=args.video_path,
        weights_path=args.weights,
        greedy=(args.decoding == "greedy"),
        beam_width=args.beam_width,
    )
