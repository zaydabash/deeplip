"""
Script to create a simple repository banner/visualization.
Creates an architecture diagram image for the README.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram(output_path="docs/architecture.png"):
    """
    Create a visual architecture diagram for the repository.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Deep Lip Reading Architecture', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Define boxes with positions and labels
    boxes = [
        (5, 10, 'Video Input\n75 frames × 46×140', '#E3F2FD'),
        (5, 8.5, 'Conv3D Block 1\n128 filters', '#BBDEFB'),
        (5, 7.5, 'MaxPool3D', '#90CAF9'),
        (5, 6.5, 'Conv3D Block 2\n256 filters', '#64B5F6'),
        (5, 5.5, 'MaxPool3D', '#42A5F5'),
        (5, 4.5, 'Conv3D Block 3\n75 filters', '#2196F3'),
        (5, 3.5, 'MaxPool3D', '#1E88E5'),
        (5, 2.5, 'TimeDistributed\nFlatten', '#1976D2'),
        (5, 1.5, 'Bidirectional LSTM\n128 units × 2', '#1565C0'),
        (5, 0.5, 'Dense + Softmax\nCTC Decoding', '#0D47A1'),
    ]
    
    # Draw boxes
    for x, y, label, color in boxes:
        box = FancyBboxPatch((x-1.2, y-0.3), 2.4, 0.6,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    # Draw arrows
    for i in range(len(boxes) - 1):
        arrow = FancyArrowPatch((5, boxes[i][1] - 0.3), (5, boxes[i+1][1] + 0.3),
                               arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
    
    # Add output label
    ax.text(7.5, 0.5, 'Predicted\nText', ha='center', va='center',
           fontsize=10, fontweight='bold', style='italic')
    arrow_out = FancyArrowPatch((6.2, 0.5), (7.5, 0.5),
                               arrowstyle='->', lw=2, color='green')
    ax.add_patch(arrow_out)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs("docs", exist_ok=True)
    create_architecture_diagram()
    print("\nTo add this image to your README, add:")
    print('  <img src="docs/architecture.png" alt="Architecture" width="800"/>')

