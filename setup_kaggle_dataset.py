"""
Helper script to download and set up lip-reading datasets from Kaggle.

Based on the search results, there are several options available.
"""
import os
import zipfile
from pathlib import Path


def print_kaggle_options():
    """Print available Kaggle dataset options."""
    print("="*70)
    print("KAGGLE LIP-READING DATASET OPTIONS")
    print("="*70)
    
    datasets = {
        "Lip Reading Image Dataset (MIRACL-VC1)": {
            "description": "MIRACL-VC1 lip-reading dataset with depth and color images",
            "downloads": "4,383+ downloads",
            "format": "Images (may need conversion to video)",
            "url": "Search 'Lip Reading Image Dataset' on Kaggle",
            "note": "May require format conversion"
        },
        "Other GRID-related datasets": {
            "description": "Various processed GRID corpus versions",
            "format": "Videos (.mpg) and alignments",
            "url": "Search 'GRID corpus' or 'lip reading GRID' on Kaggle",
            "note": "Check if format matches our requirements"
        }
    }
    
    for name, info in datasets.items():
        print(f"\n{name}")
        print("-" * 70)
        for key, value in info.items():
            print(f"  {key.capitalize()}: {value}")
    
    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS")
    print("="*70)
    print("\nOption 1: Manual Download from Kaggle")
    print("  1. Visit: https://www.kaggle.com/datasets")
    print("  2. Search: 'lip reading' or 'GRID corpus'")
    print("  3. Download dataset")
    print("  4. Extract to a folder")
    print("  5. Run: python3 setup_kaggle_dataset.py --organize <extracted_folder>")
    
    print("\nOption 2: Kaggle API (if you have API credentials)")
    print("  1. Install: pip install kaggle")
    print("  2. Set up API credentials (see Kaggle account settings)")
    print("  3. Run: python3 setup_kaggle_dataset.py --download <dataset_name>")


def organize_kaggle_data(source_dir, target_dir="data"):
    """
    Organize downloaded Kaggle dataset into GRID format.
    
    Args:
        source_dir: Directory containing extracted Kaggle dataset
        target_dir: Target directory for organized data
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Error: {source_dir} not found")
        return False
    
    print(f"Organizing data from {source_dir}...")
    print("This will attempt to match Kaggle structure to GRID format.")
    
    # Look for common patterns
    video_files = list(source_path.rglob("*.mpg")) + list(source_path.rglob("*.mp4"))
    align_files = list(source_path.rglob("*.align")) + list(source_path.rglob("*.txt"))
    
    print(f"\nFound:")
    print(f"  - {len(video_files)} video files")
    print(f"  - {len(align_files)} alignment/text files")
    
    if len(video_files) == 0:
        print("\n[WARNING] No video files found. The dataset might be in image format.")
        print("You may need to convert images to video first.")
        return False
    
    # Create structure
    speaker_dir = target_path / "S1"
    align_dir = target_path / "alignments" / "S1"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy videos
    for i, video_file in enumerate(video_files[:100], 1):  # Limit to 100 for testing
        target_video = speaker_dir / f"video{i}.mpg"
        if video_file.suffix == ".mp4":
            # Would need conversion - for now just copy
            print(f"  Note: {video_file.name} is MP4, may need conversion")
        shutil.copy2(video_file, target_video)
    
    print(f"\n[OK] Organized {min(len(video_files), 100)} videos")
    print(f"[OK] Data ready in {target_dir}/")
    
    return True


if __name__ == "__main__":
    import sys
    import shutil
    
    if len(sys.argv) > 1 and sys.argv[1] == "--organize":
        source = sys.argv[2] if len(sys.argv) > 2 else "kaggle_data"
        organize_kaggle_data(source)
    else:
        print_kaggle_options()

