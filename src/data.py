"""Video loading, alignment parsing, and dataset download/extraction.

Module-level path constants (ALIGNMENTS_DIR, DATA_URL, DATA_ZIP_PATH, DATA_DIR)
and the ``gdown`` module are imported here so they can be monkeypatched in tests
and overridden at runtime.
"""
import os
import zipfile

import cv2
import gdown
import numpy as np
import tensorflow as tf

from .config import (
    ALIGNMENTS_DIR,
    DATA_DIR,
    DATA_URL,
    DATA_ZIP_PATH,
    MOUTH_REGION,
)


def load_video(path):
    """Load a video, crop the mouth region, grayscale and standardize it.

    Returns a float32 array of shape [frames, VIDEO_HEIGHT, VIDEO_WIDTH, 1].
    Raises ValueError if no frames can be read.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = gray[
                MOUTH_REGION["top"]:MOUTH_REGION["bottom"],
                MOUTH_REGION["left"]:MOUTH_REGION["right"],
            ]
            frames.append(crop)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {path}")

    video = np.array(frames, dtype=np.float32)
    mean = video.mean()
    std = video.std()
    if std > 0:
        video = (video - mean) / std
    else:
        # Constant clip: avoid division by zero, mean-subtraction yields zeros.
        video = video - mean

    video = np.expand_dims(video, axis=-1)
    return video.astype(np.float32)


def load_alignments(path, char_to_num):
    """Parse a .align file into a tensor of character ids.

    Lines are ``start end token``; tokens equal to "silence"/"sil" (any case)
    are dropped. Remaining tokens are lowercased and joined with spaces, then
    mapped to character ids. Returns an empty int tensor for silence-only files.
    """
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            token = parts[2]
            if token.lower() in ("silence", "sil"):
                continue
            tokens.append(token.lower())

    chars = list(" ".join(tokens))
    if not chars:
        return tf.zeros([0], dtype=tf.int64)
    return char_to_num(chars)


def load_data(video_path, char_to_num):
    """Load a video and its matching alignment.

    The alignment is found at ``ALIGNMENTS_DIR/<speaker>/<stem>.align`` where the
    speaker is the video's parent directory name.
    """
    video = load_video(video_path)

    speaker = os.path.basename(os.path.dirname(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    align_path = os.path.join(ALIGNMENTS_DIR, speaker, stem + ".align")

    alignment = load_alignments(align_path, char_to_num)
    return video, alignment


def load_data_tf(video_path, char_to_num):
    """tf.data-friendly wrapper around load_data.

    Accepts a string tensor or a plain string and returns (float32 video,
    int32 alignment) tensors.
    """
    if hasattr(video_path, "numpy"):
        path = video_path.numpy()
    else:
        path = video_path
    if isinstance(path, bytes):
        path = path.decode("utf-8")

    video, alignment = load_data(path, char_to_num)
    video = tf.convert_to_tensor(video, dtype=tf.float32)
    alignment = tf.cast(alignment, tf.int32)
    return video, alignment


def download_and_extract_data():
    """Download (optionally) and extract the dataset zip into DATA_DIR.

    If DATA_URL is set, the zip is downloaded with gdown first; otherwise a
    locally present DATA_ZIP_PATH is used. Each failure mode prints a message
    rather than raising, so callers/CI can run the flow offline.
    """
    if DATA_URL:
        print(f"Downloading data from {DATA_URL}...")
        try:
            gdown.download(DATA_URL, DATA_ZIP_PATH, quiet=False)
        except Exception as exc:  # noqa: BLE001 - report any download error
            print(f"Download failed: {exc}")
            return
        if not os.path.exists(DATA_ZIP_PATH):
            print(f"Cannot extract: {DATA_ZIP_PATH} not found after download.")
            return
    else:
        if not os.path.exists(DATA_ZIP_PATH):
            print(f"Data zip not found at {DATA_ZIP_PATH}. Set DATA_URL or provide it locally.")
            return

    try:
        with zipfile.ZipFile(DATA_ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"Data extracted to {DATA_DIR}")
    except Exception as exc:  # noqa: BLE001 - report any extraction error
        print(f"Extraction failed: {exc}")
