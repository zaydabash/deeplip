"""
Helper script to adapt alternative lip-reading datasets to work with this project.

This script helps convert datasets like LRW, LRS2, etc. to the GRID format structure.
"""
import os
import shutil
from pathlib import Path


def print_dataset_info():
    """Print information about alternative datasets."""
    print("="*70)
    print("ALTERNATIVE LIP-READING DATASETS")
    print("="*70)
    
    datasets = {
        "LRW (Lip Reading in the Wild)": {
            "url": "https://www.robots.ox.ac.uk/~vgg/data/lip_reading/",
            "description": "Large-scale dataset with 500 words",
            "format": "MP4 videos with word-level labels",
            "size": "~50GB",
            "notes": "Requires conversion to GRID format"
        },
        "LRS2/LRS3": {
            "url": "https://www.robots.ox.ac.uk/~vgg/data/lrs/",
            "description": "BBC dataset with sentence-level transcriptions",
            "format": "MP4 videos with text transcriptions",
            "size": "Very large (100GB+)",
            "notes": "Requires research access request"
        },
        "MIRACL-VC1": {
            "url": "https://sites.google.com/view/miracl-vc1",
            "description": "Multimodal dataset",
            "format": "Various formats",
            "size": "Medium",
            "notes": "Check website for access"
        }
    }
    
    for name, info in datasets.items():
        print(f"\n{name}")
        print("-" * 70)
        print(f"URL: {info['url']}")
        print(f"Description: {info['description']}")
        print(f"Format: {info['format']}")
        print(f"Size: {info['size']}")
        print(f"Notes: {info['notes']}")
    
    print("\n" + "="*70)
    print("CONVERSION GUIDE")
    print("="*70)
    print("\nTo use alternative datasets, you need to:")
    print("1. Download the dataset")
    print("2. Convert videos to .mpg format (if needed)")
    print("3. Create .align files from transcriptions")
    print("4. Organize in GRID structure: data/S1/, data/alignments/S1/")
    print("\nSee convert_lrw.py for LRW-specific conversion example.")


def create_lrw_converter_template():
    """Create a template script for converting LRW dataset."""
    template = '''"""
Template script to convert LRW dataset to GRID format.

LRW dataset structure:
lrw/
├── train/
│   ├── word1/
│   │   ├── video1.mp4
│   │   └── ...
│   └── ...
└── test/...

Convert to GRID format:
data/
├── S1/
│   ├── video1.mpg
│   └── ...
└── alignments/
    └── S1/
        ├── video1.align
        └── ...
"""
import os
import cv2
from pathlib import Path


def convert_lrw_to_grid(lrw_dir, output_dir="data"):
    """
    Convert LRW dataset to GRID format.
    
    Args:
        lrw_dir: Path to LRW dataset root
        output_dir: Output directory for GRID format
    """
    lrw_path = Path(lrw_dir)
    output_path = Path(output_dir)
    
    # Create directories
    video_dir = output_path / "S1"
    align_dir = output_path / "alignments" / "S1"
    video_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    
    video_count = 0
    
    # Process each word directory
    for word_dir in lrw_path.glob("*/"):
        if not word_dir.is_dir():
            continue
        
        word = word_dir.name
        
        # Process each video
        for video_file in word_dir.glob("*.mp4"):
            # Convert MP4 to MPG
            output_video = video_dir / f"{video_file.stem}.mpg"
            convert_video(video_file, output_video)
            
            # Create alignment file
            align_file = align_dir / f"{video_file.stem}.align"
            create_alignment(align_file, word)
            
            video_count += 1
            
            if video_count % 100 == 0:
                print(f"Processed {video_count} videos...")
    
    print(f"Conversion complete! Processed {video_count} videos.")


def convert_video(input_path, output_path):
    """Convert video from MP4 to MPG format."""
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()


def create_alignment(align_path, word):
    """Create alignment file for a single word."""
    # LRW has single words, so create simple alignment
    with open(align_path, 'w') as f:
        f.write(f"0.0 0.5 silence\\n")
        f.write(f"0.5 1.5 {word}\\n")
        f.write(f"1.5 2.0 silence\\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_lrw.py <lrw_dataset_directory>")
        print("Example: python convert_lrw.py /path/to/lrw")
        sys.exit(1)
    
    lrw_dir = sys.argv[1]
    convert_lrw_to_grid(lrw_dir)
'''
    
    with open("convert_lrw.py", 'w') as f:
        f.write(template)
    
    print("Created convert_lrw.py template script.")
    print("This script can help convert LRW dataset to GRID format.")


if __name__ == "__main__":
    print_dataset_info()
    print("\n" + "="*70)
    response = input("Create LRW converter template? (y/n): ")
    if response.lower() == 'y':
        create_lrw_converter_template()

