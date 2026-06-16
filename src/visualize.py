"""Visualization utilities: save a preprocessed clip as an animated GIF."""
import imageio
import numpy as np

from . import config
from .data import load_video


def save_video_animation(video, output_path, fps=10):
    """Save a [frames, H, W, 1] float clip as a grayscale GIF.

    Pixel values are normalized to 0-255 across the whole clip; a constant clip
    (v_max == v_min) falls back to all-black frames instead of dividing by zero.
    """
    v_min = float(video.min())
    v_max = float(video.max())
    span = v_max - v_min

    frames = []
    for frame in video:
        gray = frame[..., 0]
        if span > 0:
            norm = (gray - v_min) / span
        else:
            norm = np.zeros_like(gray)
        frames.append((norm * 255).astype(np.uint8))

    # imageio expects per-frame duration in milliseconds (fps was deprecated).
    imageio.mimsave(output_path, frames, duration=1000 / fps)


def visualize_preprocessed_clip(video_path, output_path):
    """Load and preprocess a clip (pad/truncate to TARGET_FRAMES) and save a GIF."""
    video = load_video(video_path)

    if video.shape[0] > config.TARGET_FRAMES:
        video = video[:config.TARGET_FRAMES]
    elif video.shape[0] < config.TARGET_FRAMES:
        pad = np.zeros(
            (config.TARGET_FRAMES - video.shape[0],) + video.shape[1:],
            dtype=video.dtype,
        )
        video = np.concatenate([video, pad], axis=0)

    save_video_animation(video, output_path)
