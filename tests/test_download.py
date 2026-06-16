"""Tests for src.data.download_and_extract_data: branch coverage via monkeypatching.

No real network access or large downloads happen here - gdown and the
filesystem are monkeypatched so each branch of the download/extract flow
can be exercised quickly and offline.
"""
import zipfile

from src.data import download_and_extract_data


def test_skips_download_and_reports_missing_zip_when_url_is_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.data.DATA_URL", None)
    monkeypatch.setattr("src.data.DATA_ZIP_PATH", str(tmp_path / "missing.zip"))

    download_and_extract_data()

    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_extracts_local_zip_when_url_is_none_and_zip_present(tmp_path, monkeypatch, capsys):
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("sample.txt", "hello")

    extract_dir = tmp_path / "extracted"
    monkeypatch.setattr("src.data.DATA_URL", None)
    monkeypatch.setattr("src.data.DATA_ZIP_PATH", str(zip_path))
    monkeypatch.setattr("src.data.DATA_DIR", str(extract_dir))

    download_and_extract_data()

    captured = capsys.readouterr()
    assert "Data extracted to" in captured.out
    assert (extract_dir / "sample.txt").exists()


def test_reports_download_failure_when_gdown_raises(tmp_path, monkeypatch, capsys):
    def _raise_download(*args, **kwargs):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr("src.data.DATA_URL", "https://example.com/data.zip")
    monkeypatch.setattr("src.data.DATA_ZIP_PATH", str(tmp_path / "data.zip"))
    monkeypatch.setattr("src.data.gdown.download", _raise_download)

    download_and_extract_data()

    captured = capsys.readouterr()
    assert "Download failed" in captured.out


def test_reports_missing_zip_after_successful_download_call(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.data.DATA_URL", "https://example.com/data.zip")
    monkeypatch.setattr("src.data.DATA_ZIP_PATH", str(tmp_path / "data.zip"))
    monkeypatch.setattr("src.data.gdown.download", lambda *a, **k: None)

    download_and_extract_data()

    captured = capsys.readouterr()
    assert "Cannot extract" in captured.out


def test_reports_extraction_failure_for_invalid_zip(tmp_path, monkeypatch, capsys):
    bad_zip = tmp_path / "data.zip"
    bad_zip.write_text("not a real zip")

    monkeypatch.setattr("src.data.DATA_URL", None)
    monkeypatch.setattr("src.data.DATA_ZIP_PATH", str(bad_zip))
    monkeypatch.setattr("src.data.DATA_DIR", str(tmp_path / "extracted"))

    download_and_extract_data()

    captured = capsys.readouterr()
    assert "Extraction failed" in captured.out
