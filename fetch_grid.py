"""Resumable, self-retrying downloader for the GRID corpus from Zenodo.

Designed for slow/flaky connections: each file is fetched with curl's resume
(-C -) and retry options, and the whole curl invocation is repeated until the
local file size matches the size Zenodo reports. Safe to re-run at any time; it
skips files that are already complete and resumes partial ones.

Usage:
    python fetch_grid.py                 # alignments + all 34 speakers
    python fetch_grid.py --audio         # also download audio_25k.zip
    python fetch_grid.py --speakers 1 2  # only specific speakers (+ alignments)
"""
import argparse
import os
import subprocess
import sys

import requests

RECORD = "3625687"
API_URL = f"https://zenodo.org/api/records/{RECORD}"
DOWNLOAD_DIR = "downloads"
MAX_ATTEMPTS = 500


def _download_url(key):
    return f"https://zenodo.org/records/{RECORD}/files/{key}?download=1"


def _order(key):
    if key == "alignments.zip":
        return (0, 0)
    if key.startswith("s") and key.endswith(".zip"):
        try:
            return (1, int(key[1:-4]))
        except ValueError:
            return (2, key)
    return (3, key)


def fetch_file(key, expected_size):
    dest = os.path.join(DOWNLOAD_DIR, key)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have == expected_size:
            print(f"[ok] {key} complete ({expected_size:,} bytes)", flush=True)
            return True
        if have > expected_size:
            # Corrupt/oversized partial: start over.
            os.remove(dest)
            have = 0
        pct = (have / expected_size * 100) if expected_size else 0
        print(
            f"[dl] {key} attempt {attempt}: have {have:,}/{expected_size:,} ({pct:.1f}%)",
            flush=True,
        )
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry", "100",
                "--retry-delay", "5",
                "--retry-all-errors",
                "--retry-connrefused",
                "--connect-timeout", "60",
                "--no-progress-meter",
                "-C", "-",
                "-o", dest,
                _download_url(key),
            ]
        )
    print(f"[FAIL] {key} did not complete after {MAX_ATTEMPTS} attempts", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description="Resumable GRID corpus downloader.")
    parser.add_argument("--audio", action="store_true", help="Also download audio_25k.zip (~2.6 GB).")
    parser.add_argument("--speakers", nargs="*", type=int, help="Only these speaker numbers (plus alignments).")
    args = parser.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    files = {f["key"]: f["size"] for f in requests.get(API_URL, timeout=60).json()["files"]}

    if args.speakers:
        wanted = ["alignments.zip"] + [f"s{n}.zip" for n in args.speakers]
    else:
        wanted = [k for k in files if k.endswith(".zip") and k != "audio_25k.zip"]
        if args.audio:
            wanted.append("audio_25k.zip")

    wanted = [k for k in wanted if k in files]
    wanted.sort(key=_order)

    total = sum(files[k] for k in wanted)
    print(f"Downloading {len(wanted)} files, ~{total / 1e9:.2f} GB total, into {DOWNLOAD_DIR}/", flush=True)

    ok = True
    for key in wanted:
        ok = fetch_file(key, files[key]) and ok

    if ok:
        print("ALL DONE - all requested files complete.", flush=True)
        print("Next: python download_grid_zenodo.py --extract downloads/", flush=True)
    else:
        print("SOME FILES FAILED - re-run this script to resume.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
