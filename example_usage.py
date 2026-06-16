"""
Example usage script demonstrating how to use the lip-reading project.
"""
import os
from src.data import download_and_extract_data
from src.visualize import visualize_preprocessed_clip
from src.predict import predict_clip, load_model
from src.dataset import build_vocab_lookup

# Example 1: Download data
# Uncomment to download data:
# download_and_extract_data()

# Example 2: Visualize a preprocessed clip
# Uncomment to create an animation GIF:
# visualize_preprocessed_clip("data/S1/video1.mpg", "example_animation.gif")

# Example 3: Run prediction
# Uncomment to predict text from a video:
# _, num_to_char = build_vocab_lookup()
# model = load_model("models/weights_epoch_96.h5")
# predicted_text = predict_clip("data/S1/video1.mpg", model, num_to_char)
# print(f"Predicted text: {predicted_text}")

print("Example usage script loaded. Uncomment the examples above to use.")

