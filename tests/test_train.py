"""Tests for src.train: GPU setup helper and the end-to-end training loop."""
from src.train import main as train_main
from src.train import setup_gpu
from tests.helpers import write_align_file, write_test_video


def test_setup_gpu_runs_without_error(capsys):
    setup_gpu()

    captured = capsys.readouterr()
    assert "GPU" in captured.out


def test_main_trains_one_epoch_on_synthetic_data(tmp_path, monkeypatch, capsys):
    """End-to-end smoke test: build a tiny dataset, run one training epoch,
    and confirm a checkpoint comes out the other side.

    Uses 4 synthetic clips so prepare_dataset() produces 2 batches (BATCH_SIZE=2):
    one batch for training, one for validation - enough to exercise the full
    train/validation/checkpoint/callback pipeline without a real dataset.
    """
    speaker_dir = tmp_path / "data" / "S1"
    align_dir = tmp_path / "data" / "alignments" / "S1"
    speaker_dir.mkdir(parents=True)
    align_dir.mkdir(parents=True)

    words = ["one", "two", "three", "four"]
    for i, word in enumerate(words):
        write_test_video(speaker_dir / f"video{i}.mp4", num_frames=10)
        write_align_file(align_dir / f"video{i}.align", [f"0.0 1.0 {word}"])

    models_dir = tmp_path / "models"
    monkeypatch.setattr("src.data.ALIGNMENTS_DIR", str(tmp_path / "data" / "alignments"))
    monkeypatch.setattr("src.train.TRAIN_SIZE", 1)
    monkeypatch.setattr("src.train.MODEL_SAVE_DIR", str(models_dir))

    train_main(video_pattern=str(speaker_dir / "*.mp4"), epochs=1)

    captured = capsys.readouterr()
    assert "Training completed!" in captured.out
    assert (models_dir / "weights_epoch_01.h5").exists()
