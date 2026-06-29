"""Configuration and hyperparameters for the lip-reading pipeline.

Every invariant the rest of the codebase relies on lives here so that the
model, dataset, loss and callbacks stay in sync (see tests/test_config.py).
"""

# --- Vocabulary -------------------------------------------------------------
# Lowercase letters, digits and a single space. StringLookup maps these to ids
# 1..VOCAB_SIZE (id 0 is reserved for padding/mask), and the CTC blank lives at
# BLANK_TOKEN = VOCAB_SIZE + 1, one past the last real character id so it can
# never collide with a character.
VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789 "
VOCAB_SIZE = len(VOCAB)
BLANK_TOKEN = VOCAB_SIZE + 1

# --- Video preprocessing ----------------------------------------------------
# Mouth region (in pixels) cropped from each frame. The model input dimensions
# are derived from this region so they always match.
MOUTH_REGION = {"top": 190, "bottom": 236, "left": 100, "right": 240}
VIDEO_HEIGHT = MOUTH_REGION["bottom"] - MOUTH_REGION["top"]   # 46
VIDEO_WIDTH = MOUTH_REGION["right"] - MOUTH_REGION["left"]    # 140

TARGET_FRAMES = 75      # clips are padded/truncated to this many frames
MAX_TEXT_LENGTH = 40    # alignment sequences are padded/truncated to this length

# --- Model architecture -----------------------------------------------------
CONV3D_FILTERS = [32, 64, 128]
LSTM_UNITS = 128
DROPOUT_RATE = 0.5

# --- Training ---------------------------------------------------------------
BATCH_SIZE = 2
EPOCHS = 100
INITIAL_LEARNING_RATE = 1e-4
LR_DECAY_START_EPOCH = 30   # constant LR before this epoch, exponential decay after
LR_DECAY_RATE = 0.9
TRAIN_SIZE = 450            # number of clips used for training
VAL_SIZE = 50               # number of clips (after the training split) used for validation

# --- Paths ------------------------------------------------------------------
DATA_DIR = "data"
ALIGNMENTS_DIR = "data/alignments"
DATA_ZIP_PATH = "data.zip"
# Optional Google Drive URL for download_and_extract_data(); None means use a
# locally provided DATA_ZIP_PATH instead of downloading.
DATA_URL = None
MODEL_SAVE_DIR = "models"
DEFAULT_WEIGHTS = "models/weights_epoch_01.h5"

# --- Prediction ---------------------------------------------------------
BEAM_WIDTH = 100   # search width used by beam-search CTC decoding
