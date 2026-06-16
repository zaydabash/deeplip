"""Tests for src.visualize: saving preprocessed clips as GIF animations."""
import numpy as np

from src import config
from src.visualize import save_video_animation, visualize_preprocessed_clip
from tests.helpers import write_test_video


def test_save_video_animation_writes_gif(tmp_path):
    video = np.random.rand(5, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1).astype(np.float32)
    output_path = tmp_path / "out.gif"

    save_video_animation(video, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_video_animation_handles_constant_video(tmp_path):
    # All-zero video has v_max == v_min, exercising the fallback normalization branch.
    video = np.zeros((5, config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 1), dtype=np.float32)
    output_path = tmp_path / "constant.gif"

    save_video_animation(video, str(output_path))

    assert output_path.exists()


def test_visualize_preprocessed_clip_pads_and_saves(tmp_path):
    video_path = tmp_path / "clip.mp4"
    write_test_video(video_path, num_frames=5)
    output_path = tmp_path / "anim.gif"

    visualize_preprocessed_clip(str(video_path), str(output_path))

    assert output_path.exists()


def test_visualize_preprocessed_clip_truncates_long_clip(tmp_path):
    video_path = tmp_path / "long_clip.mp4"
    write_test_video(video_path, num_frames=config.TARGET_FRAMES + 5)
    output_path = tmp_path / "anim_long.gif"

    visualize_preprocessed_clip(str(video_path), str(output_path))

    assert output_path.exists()
