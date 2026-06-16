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


def decode_predictions(predictions, num_to_char):
    """Greedy-decode model predictions ([1, time, classes] or [time, classes])."""
    pred = np.asarray(predictions)
    if pred.ndim == 3:
        pred = pred[0]
    ids = _greedy_decode_ids(pred)
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


def predict_clip(video_path, model, num_to_char):
    """Preprocess a video, run the model, and return the decoded text."""
    video = load_video(video_path)
    video = pad_video(video)
    video = np.expand_dims(video, axis=0)
    predictions = model.predict(video, verbose=0)
    return decode_predictions(predictions, num_to_char)


def main(video_path, weights_path=config.DEFAULT_WEIGHTS):
    """Run end-to-end prediction for a single video and print the result."""
    _, num_to_char = build_vocab_lookup()
    model = load_model(weights_path)
    text = predict_clip(video_path, model, num_to_char)

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
    args = parser.parse_args()
    main(video_path=args.video_path, weights_path=args.weights)
