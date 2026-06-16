"""
Extract and organize the archive (2).zip file.
"""
import os
import zipfile
import shutil
from pathlib import Path


def extract_and_organize(archive_path, output_dir="data"):
    """
    Extract archive and organize into GRID format structure.
    
    Args:
        archive_path: Path to the zip file
        output_dir: Output directory
    """
    archive = Path(archive_path)
    output = Path(output_dir)
    
    if not archive.exists():
        print(f"Error: {archive_path} not found")
        return False
    
    print(f"Extracting {archive.name} ({archive.stat().st_size / (1024**3):.1f} GB)...")
    print("This may take a few minutes...")
    
    # Create temp extraction directory
    temp_dir = Path("temp_extract")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Extract archive
        print("Extracting archive...")
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        print("[OK] Extraction complete")
        print("\nAnalyzing structure...")
        
        # Find videos and alignments
        video_files = list(temp_dir.rglob("*.mpg")) + list(temp_dir.rglob("*.mp4"))
        align_files = list(temp_dir.rglob("*.align")) + list(temp_dir.rglob("*.txt"))
        
        print(f"Found:")
        print(f"  - {len(video_files)} video files")
        print(f"  - {len(align_files)} alignment/text files")
        
        # Check for common structures
        # Look for speaker directories (S1, S2, etc.)
        speaker_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and (d.name.startswith('S') or d.name.isdigit())]
        
        if speaker_dirs:
            print(f"\nFound {len(speaker_dirs)} speaker directories")
            # Copy speaker directories
            for speaker_dir in speaker_dirs:
                target = output / speaker_dir.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(speaker_dir, target)
                print(f"  [OK] Copied {speaker_dir.name}/")
        
        # Look for alignments directory
        alignments_source = None
        for possible in [temp_dir / "alignments", temp_dir / "alignment", temp_dir / "align"]:
            if possible.exists():
                alignments_source = possible
                break
        
        if alignments_source:
            target_align = output / "alignments"
            if target_align.exists():
                shutil.rmtree(target_align)
            shutil.copytree(alignments_source, target_align)
            print(f"  [OK] Copied alignments/")
        elif align_files:
            # Create alignments structure
            align_dir = output / "alignments"
            align_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [NOTE] Creating alignments structure manually")
        
        # If videos are in root, organize them
        if video_files and not speaker_dirs:
            speaker_dir = output / "S1"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            for i, video in enumerate(video_files[:100], 1):  # Limit for testing
                target = speaker_dir / f"video{i}{video.suffix}"
                shutil.copy2(video, target)
            print(f"  [OK] Organized videos into S1/")
        
        print(f"\n{'='*70}")
        print("EXTRACTION COMPLETE!")
        print(f"{'='*70}")
        print(f"\nData organized in: {output_dir}/")
        print("\nNext steps:")
        print("1. Verify structure: ls -la data/")
        print("2. Check for videos: ls data/S*/")
        print("3. Start training: python3 -m src.train --video_pattern 'data/S*/*.mpg' --epochs 5")
        
        return True
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            print("\nCleaning up temporary files...")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import sys
    
    archive_path = "~/Downloads/archive (2).zip"
    if len(sys.argv) > 1:
        archive_path = sys.argv[1]
    
    extract_and_organize(archive_path)

