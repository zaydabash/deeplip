"""
Helper script to obtain or prepare data.zip for the lip-reading project.

This script provides options to:
1. Download from Google Drive (if URL is provided)
2. Create data.zip from existing raw data files
3. Provide instructions for obtaining GRID dataset
"""
import os
import zipfile
from pathlib import Path


def print_data_instructions():
    """Print instructions for obtaining the GRID dataset."""
    print("="*70)
    print("HOW TO OBTAIN DATA FOR LIP-READING PROJECT")
    print("="*70)
    print("\nOption 1: Download GRID Dataset (Official)")
    print("-" * 70)
    print("The GRID corpus is a lip-reading dataset.")
    print("1. Visit: https://www.dcs.shef.ac.uk/spandh/gridcorpus/")
    print("2. Request access to the GRID corpus")
    print("3. Download videos and alignments")
    print("4. Organize them in the required structure (see below)")
    print("\nOption 2: Use Pre-prepared data.zip")
    print("-" * 70)
    print("If you have a Google Drive link with data.zip:")
    print("1. Update DATA_URL in src/config.py")
    print("2. Run: python3 -c \"from src.data import download_and_extract_data; download_and_extract_data()\"")
    print("\nOption 3: Create data.zip from Existing Files")
    print("-" * 70)
    print("If you already have videos and alignments:")
    print("1. Organize them in this structure:")
    print("   data/")
    print("   ├── S1/")
    print("   │   ├── video1.mpg")
    print("   │   ├── video2.mpg")
    print("   │   └── ...")
    print("   └── alignments/")
    print("       └── S1/")
    print("           ├── video1.align")
    print("           ├── video2.align")
    print("           └── ...")
    print("2. Run: python3 get_data.py --create-zip")
    print("\n" + "="*70)


def create_data_zip_from_existing():
    """Create data.zip from existing data/ directory."""
    data_dir = "data"
    zip_path = "data.zip"
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir}/ directory not found.")
        print("Please create the data directory with videos and alignments first.")
        return False
    
    print(f"Creating {zip_path} from {data_dir}/...")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, os.path.dirname(data_dir))
                    zipf.write(file_path, arc_name)
                    print(f"  Added: {arc_name}")
        
        file_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
        print(f"\nSuccess! Created {zip_path} ({file_size:.1f} MB)")
        return True
    except Exception as e:
        print(f"Error creating zip: {e}")
        return False


def check_data_structure():
    """Check if data directory has the correct structure."""
    data_dir = Path("data")
    
    if not data_dir.exists():
        return False, "data/ directory does not exist"
    
    # Check for video directories (e.g., S1, S2, etc.)
    video_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name == "alignments"]
    if not video_dirs:
        return False, "No video directories found (expected S1/, S2/, etc.)"
    
    # Check for alignments directory
    alignments_dir = data_dir / "alignments"
    if not alignments_dir.exists():
        return False, "alignments/ directory not found"
    
    # Check for video files
    video_files = list(data_dir.glob("S*/*.mpg"))
    if not video_files:
        return False, "No .mpg video files found"
    
    # Check for alignment files
    align_files = list(alignments_dir.glob("S*/*.align"))
    if not align_files:
        return False, "No .align files found"
    
    return True, f"Found {len(video_files)} videos and {len(align_files)} alignments"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-zip":
        # Check structure first
        valid, message = check_data_structure()
        if not valid:
            print(f"Error: {message}")
            print("\nPlease ensure your data/ directory has the correct structure.")
            print_data_instructions()
            sys.exit(1)
        
        print(message)
        print()
        if create_data_zip_from_existing():
            print("\nYou can now use data.zip for the project!")
        else:
            sys.exit(1)
    else:
        # Print instructions
        print_data_instructions()
        
        # Check if data directory exists
        if os.path.exists("data"):
            print("\nChecking existing data/ directory...")
            valid, message = check_data_structure()
            print(f"Status: {message}")
            if valid:
                print("\nYou can create data.zip by running:")
                print("  python3 get_data.py --create-zip")
        else:
            print("\nNo data/ directory found.")
            print("Follow one of the options above to obtain the data.")

