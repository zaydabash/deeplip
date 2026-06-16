"""
Quick script to check training status and view latest results.
"""
import os
from pathlib import Path

models_dir = Path("models")

if models_dir.exists():
    weight_files = sorted(models_dir.glob("weights_epoch_*.h5"))
    
    if weight_files:
        latest = weight_files[-1]
        epoch_num = latest.stem.split("_")[-1]
        
        print("="*60)
        print("TRAINING STATUS")
        print("="*60)
        print(f"\nTraining completed!")
        print(f"Latest epoch: {epoch_num}")
        print(f"Latest weights: {latest.name}")
        print(f"File size: {latest.stat().st_size / (1024*1024):.1f} MB")
        print(f"\nTotal checkpoints: {len(weight_files)}")
        print("\nAll saved weights:")
        for wf in weight_files[-5:]:  # Show last 5
            print(f"  - {wf.name}")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Continue training:")
        print("   python3 -m src.train --epochs 20")
        print("\n2. Test predictions:")
        print("   python3 -m src.predict data/S1/video1.mp4 --weights models/weights_epoch_10.h5")
        print("\n3. Visualize preprocessed video:")
        print("   python3 -c \"from src.visualize import visualize_preprocessed_clip; visualize_preprocessed_clip('data/S1/video1.mp4', 'test.gif')\"")
    else:
        print("No weights found. Start training with:")
        print("  python3 -m src.train --epochs 5")
else:
    print("Models directory not found. Start training first.")

