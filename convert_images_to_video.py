"""
Convert image sequences to video files for lip-reading dataset.

This dataset appears to be image-based (MIRACL-VC1 format).
We need to convert image sequences to video files.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def convert_images_to_videos(source_dir, output_dir="data", fps=25):
    """
    Convert image sequences to video files.
    
    Args:
        source_dir: Directory containing image sequences
        output_dir: Output directory for videos
        fps: Frames per second for output videos
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    # Find all image sequences
    # Structure appears to be: cropped/cropped/F01/words/01/01/color_*.jpg
    image_files = list(source.rglob("color_*.jpg"))
    
    print(f"Found {len(image_files)} image files")
    
    # Group images by sequence (same directory)
    sequences = defaultdict(list)
    for img_file in image_files:
        seq_key = img_file.parent
        sequences[seq_key].append(img_file)
    
    print(f"Found {len(sequences)} image sequences")
    
    # Create output structure
    speaker_dir = output / "S1"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each sequence to video
    video_count = 0
    for seq_path, images in list(sequences.items())[:100]:  # Limit for testing
        # Sort images by number
        images.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        if len(images) == 0:
            continue
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(images[0]))
        if first_img is None:
            continue
        
        height, width = first_img.shape[:2]
        
        # Create video filename
        # Use path structure to create unique name
        path_parts = seq_path.parts
        video_name = "_".join([p for p in path_parts if p not in ['cropped', 'words']])[:50]
        video_path = speaker_dir / f"video{video_count+1}.mpg"
        
        # Create video writer (use MP4V codec for better compatibility)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = speaker_dir / f"video{video_count+1}.mp4"  # Use .mp4 extension
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Write frames
        for img_file in images:
            img = cv2.imread(str(img_file))
            if img is not None:
                # Resize if needed
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                out.write(img)
        
        out.release()
        video_count += 1
        
        if video_count % 10 == 0:
            print(f"  Converted {video_count} videos...")
    
    print(f"\n[OK] Converted {video_count} image sequences to videos")
    print(f"[OK] Videos saved to {speaker_dir}/")
    
    return video_count


def create_alignments_from_structure(source_dir, output_dir="data"):
    """
    Create alignment files based on image sequence structure.
    
    Args:
        source_dir: Directory containing image sequences
        output_dir: Output directory
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    align_dir = output / "alignments" / "S1"
    align_dir.mkdir(parents=True, exist_ok=True)
    
    # Find word directories
    word_dirs = list(source.rglob("words/*/*"))
    
    video_count = 0
    for word_dir in word_dirs[:100]:  # Limit for testing
        if not word_dir.is_dir():
            continue
        
        # Extract word from path
        word = word_dir.name
        
        # Create alignment file
        align_file = align_dir / f"video{video_count+1}.align"
        
        with open(align_file, 'w') as f:
            # Simple alignment: word spans the entire sequence
            f.write(f"0.0 0.5 silence\n")
            f.write(f"0.5 1.5 {word}\n")
            f.write(f"1.5 2.0 silence\n")
        
        video_count += 1
    
    print(f"[OK] Created {video_count} alignment files")
    return video_count


if __name__ == "__main__":
    import sys
    
    source = "~/Downloads/archive (2)"
    if len(sys.argv) > 1:
        source = sys.argv[1]
    
    print("="*70)
    print("CONVERTING IMAGE DATASET TO VIDEO FORMAT")
    print("="*70)
    print(f"\nSource: {source}")
    print("This may take several minutes...\n")
    
    # Convert images to videos
    video_count = convert_images_to_videos(source)
    
    # Create alignments
    align_count = create_alignments_from_structure(source)
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nCreated:")
    print(f"  - {video_count} video files")
    print(f"  - {align_count} alignment files")
    print("\nNote: Videos are in MP4 format. You may need to convert to MPG or update the code to accept MP4.")
    print("Next step: python3 -m src.train --video_pattern 'data/S1/*.mp4' --epochs 5")

