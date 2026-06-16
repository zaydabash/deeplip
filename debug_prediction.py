"""
Debug script to check what the model is actually predicting.
"""
import numpy as np
import tensorflow as tf
from src.data import load_video
from src.dataset import build_vocab_lookup
from src.model import build_model
from src.config import TARGET_FRAMES

# Load model
print("Loading model...")
_, num_to_char = build_vocab_lookup()
model = build_model()
model.load_weights("models/weights_epoch_10.h5")

# Load a video
print("Loading video...")
video = load_video("data/S1/video1.mp4")

# Pad to target frames
current_frames = video.shape[0]
if current_frames > TARGET_FRAMES:
    video = video[:TARGET_FRAMES]
elif current_frames < TARGET_FRAMES:
    padding = np.zeros((TARGET_FRAMES - current_frames, video.shape[1], video.shape[2], 1), dtype=video.dtype)
    video = np.concatenate([video, padding], axis=0)

video = np.expand_dims(video, axis=0)
print(f"Video shape: {video.shape}")

# Get predictions
print("Running prediction...")
predictions = model.predict(video, verbose=0)
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions min/max: {predictions.min():.4f} / {predictions.max():.4f}")

# Check what the model is predicting (argmax per timestep)
predicted_ids = np.argmax(predictions[0], axis=1)
print(f"\nPredicted IDs (first 20): {predicted_ids[:20]}")
print(f"Unique predicted IDs: {np.unique(predicted_ids)}")
print(f"Most common ID: {np.bincount(predicted_ids).argmax()}")

# Try CTC decode
input_length = np.array([TARGET_FRAMES])
decoded, log_probs = tf.keras.backend.ctc_decode(
    predictions,
    input_length,
    greedy=True
)

decoded_ids = decoded[0].numpy()[0]
print(f"\nCTC Decoded IDs: {decoded_ids}")
print(f"Decoded IDs (non-negative): {decoded_ids[decoded_ids >= 0]}")

# Try to convert to text
if len(decoded_ids[decoded_ids >= 0]) > 0:
    valid_ids = decoded_ids[decoded_ids >= 0]
    try:
        chars = num_to_char(tf.constant(valid_ids))
        text = ''.join([c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in chars.numpy()])
        print(f"Decoded text: '{text}'")
    except Exception as e:
        print(f"Error decoding: {e}")

print("\nNote: Empty predictions are normal early in training.")
print("The model needs more epochs to learn meaningful outputs.")

