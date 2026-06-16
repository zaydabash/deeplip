"""
Create a minimal test dataset structure for testing the lip-reading pipeline.

This creates the directory structure and sample alignment files.
You'll need to add actual video files (.mpg) manually.
"""
import os
from pathlib import Path


def create_test_structure():
    """Create minimal test dataset structure."""
    base_dir = Path("data")
    
    # Create directories
    speaker_dir = base_dir / "S1"
    alignments_dir = base_dir / "alignments" / "S1"
    
    speaker_dir.mkdir(parents=True, exist_ok=True)
    alignments_dir.mkdir(parents=True, exist_ok=True)
    
    print("Created directory structure:")
    print(f"  {speaker_dir}/")
    print(f"  {alignments_dir}/")
    
    # Create sample alignment files
    sample_alignments = [
        ("video1", "hello world"),
        ("video2", "good morning"),
        ("video3", "thank you"),
    ]
    
    for video_name, text in sample_alignments:
        align_file = alignments_dir / f"{video_name}.align"
        with open(align_file, 'w') as f:
            # Create simple alignment: split text into words
            words = text.split()
            time_per_word = 0.5
            current_time = 0.0
            
            for word in words:
                start = current_time
                end = current_time + time_per_word
                f.write(f"{start:.2f} {end:.2f} {word}\n")
                current_time = end
            
            # Add silence at end
            f.write(f"{current_time:.2f} {current_time + 0.5:.2f} silence\n")
        
        print(f"  Created: {align_file}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Add video files (.mpg format) to data/S1/")
    print("   - Name them: video1.mpg, video2.mpg, video3.mpg")
    print("   - Or update alignment filenames to match your videos")
    print("\n2. Video requirements:")
    print("   - Format: MPEG (.mpg)")
    print("   - Should contain face/mouth region")
    print("   - Can be short clips (a few seconds)")
    print("\n3. For testing, you can:")
    print("   - Use any short video files")
    print("   - Or create synthetic videos")
    print("   - Or download sample videos from the web")
    print("\n4. Once videos are added, run:")
    print("   python3 get_data.py --create-zip")
    print("="*60)


if __name__ == "__main__":
    create_test_structure()

