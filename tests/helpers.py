"""Shared helpers for generating small synthetic fixtures used across tests."""
import cv2
import numpy as np


def write_test_video(path, num_frames=5, frame_size=(240, 240), frame_value=None):
    """Write a tiny synthetic video file, large enough to crop MOUTH_REGION from.

    By default each frame gets a different gray level (so the clip has nonzero
    standard deviation). Pass frame_value to give every frame the same level
    instead (zero standard deviation).
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, frame_size)
    for i in range(num_frames):
        value = i * 10 if frame_value is None else frame_value
        frame = np.full((frame_size[1], frame_size[0], 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def write_align_file(path, lines):
    path.write_text("\n".join(lines) + "\n")
