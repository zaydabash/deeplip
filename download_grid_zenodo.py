"""
Download GRID corpus from Zenodo repository.

The GRID corpus is available at:
https://zenodo.org/records/3625687

This script helps download and organize the dataset.
"""
import os
import requests
import zipfile
from pathlib import Path


ZENODO_URL = "https://zenodo.org/records/3625687"
ZENODO_DOI = "10.5281/zenodo.3625687"

# File IDs from Zenodo (these may need to be updated - check the Zenodo page)
FILES = {
    "alignments.zip": "alignments.zip",
    "audio_25k.zip": "audio_25k.zip",
    # Speaker videos: s1.zip, s2.zip, etc. (34 speakers)
    "speakers": [f"s{i}.zip" for i in range(1, 35)]
}


def print_download_instructions():
    """Print instructions for downloading GRID from Zenodo."""
    print("="*70)
    print("DOWNLOAD GRID CORPUS FROM ZENODO")
    print("="*70)
    print(f"\nDataset URL: {ZENODO_URL}")
    print(f"DOI: {ZENODO_DOI}")
    print("\n" + "-"*70)
    print("MANUAL DOWNLOAD STEPS:")
    print("-"*70)
    print("\n1. Visit the Zenodo page:")
    print(f"   {ZENODO_URL}")
    print("\n2. Download the following files:")
    print("   - alignments.zip (word-level time alignments)")
    print("   - audio_25k.zip (audio recordings - optional)")
    print("   - s1.zip, s2.zip, ..., s34.zip (video files for each speaker)")
    print("\n3. Place all downloaded zip files in a directory (e.g., 'downloads/')")
    print("\n4. Run this script to extract and organize:")
    print("   python3 download_grid_zenodo.py --extract downloads/")
    print("\n" + "="*70)
    print("AUTOMATED DOWNLOAD (if Zenodo API allows):")
    print("="*70)
    print("\nNote: Zenodo may require manual download through the web interface.")
    print("Check the Zenodo page for direct download links.")
    print("\n" + "="*70)


def extract_and_organize(download_dir="downloads", output_dir="data"):
    """
    Extract downloaded GRID files and organize into GRID format.
    
    Args:
        download_dir: Directory containing downloaded zip files
        output_dir: Output directory for organized data
    """
    download_path = Path(download_dir)
    output_path = Path(output_dir)
    
    if not download_path.exists():
        print(f"Error: {download_dir} directory not found.")
        print("Please download files from Zenodo first.")
        return False
    
    print(f"Extracting files from {download_dir}...")
    
    # Extract alignments
    alignments_zip = download_path / "alignments.zip"
    if alignments_zip.exists():
        print("Extracting alignments...")
        with zipfile.ZipFile(alignments_zip, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print("  [OK] Alignments extracted")
    else:
        print("  [WARNING] alignments.zip not found")
    
    # Extract speaker videos
    speaker_count = 0
    for i in range(1, 35):
        speaker_zip = download_path / f"s{i}.zip"
        if speaker_zip.exists():
            print(f"Extracting speaker {i}...")
            speaker_dir = output_path / f"S{i}"
            with zipfile.ZipFile(speaker_zip, 'r') as zip_ref:
                zip_ref.extractall(speaker_dir)
            speaker_count += 1
    
    if speaker_count > 0:
        print(f"  [OK] Extracted {speaker_count} speakers")
    else:
        print("  [WARNING] No speaker zip files found")
    
    # Check structure
    if (output_path / "alignments").exists():
        print("\n" + "="*70)
        print("EXTRACTION COMPLETE!")
        print("="*70)
        print(f"\nData organized in: {output_dir}/")
        print("\nNext steps:")
        print("1. Verify structure: data/S1/, data/S2/, ..., data/alignments/")
        print("2. Create data.zip: python3 get_data.py --create-zip")
        print("3. Start training: python3 -m src.train")
        return True
    else:
        print("\n[WARNING] Structure may be incomplete.")
        print("Please check the extracted files manually.")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        download_dir = sys.argv[2] if len(sys.argv) > 2 else "downloads"
        extract_and_organize(download_dir)
    else:
        print_download_instructions()
        print("\nTo extract downloaded files, run:")
        print("  python3 download_grid_zenodo.py --extract <download_directory>")

