#!/bin/bash
# Monitor training progress and notify when complete

echo "Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    # Check if training process is running
    if ! ps aux | grep -q "[p]ython3 -m src.train"; then
        echo ""
        echo "=========================================="
        echo "TRAINING COMPLETE!"
        echo "=========================================="
        echo ""
        
        # Show latest weights
        latest=$(ls -t models/weights_epoch_*.h5 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "Latest weights: $latest"
            echo "File size: $(ls -lh $latest | awk '{print $5}')"
        fi
        
        # Show final training stats if log exists
        if [ -f training_log.txt ]; then
            echo ""
            echo "Final training stats:"
            tail -30 training_log.txt | grep -E "(Epoch|loss|Training completed)" | tail -5
        fi
        
        echo ""
        echo "Next steps:"
        echo "1. Test predictions: python3 -m src.predict data/S1/video1.mp4 --weights $latest"
        echo "2. Check status: python3 check_training_status.py"
        echo ""
        break
    fi
    
    # Show current progress
    latest_epoch=$(ls -t models/weights_epoch_*.h5 2>/dev/null | head -1 | sed 's/.*epoch_\([0-9]*\)\.h5/\1/')
    if [ -n "$latest_epoch" ]; then
        echo -ne "\rCurrent epoch: $latest_epoch/50 (or more) - $(date '+%H:%M:%S')"
    else
        echo -ne "\rWaiting for first epoch... - $(date '+%H:%M:%S')"
    fi
    
    sleep 30  # Check every 30 seconds
done

