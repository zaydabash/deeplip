# Project Status

## Code: implemented and tested

- **Data pipeline** (`src/data.py`, `src/dataset.py`): video loading, mouth-region cropping,
  alignment parsing, `tf.data` batching/padding.
- **Model** (`src/model.py`): Conv3D + BiLSTM + CTC, ~15M parameters.
- **Training** (`src/train.py`, `src/losses.py`, `src/callbacks.py`): CTC loss, learning-rate
  schedule, checkpointing, example-prediction callback.
- **Prediction** (`src/predict.py`): loads a checkpoint and decodes a video to text.
- **Visualization** (`src/visualize.py`): saves a preprocessed clip as a GIF.

All of the above is covered by the pytest suite under `tests/` (51 tests, 93% coverage of
`src/`, run in CI via `.github/workflows/ci.yml`). The suite uses small synthetic
videos/alignments generated on the fly, including an end-to-end test that runs one real
training epoch through `src.train.main()` and checks a checkpoint is written.

## Local checkpoints and data: stale, need regenerating

This repo's local (gitignored) `models/` and `data/` directories are **not usable as-is** and
should not be relied on:

1. **`models/weights_epoch_01.h5` through `weights_epoch_50.h5` are architecturally
   incompatible.** They were saved before a CTC bug fix changed the model's output layer from
   `vocab_size + 1` to `vocab_size + 2` units (a dedicated CTC blank index, see `src/model.py`
   and `src/config.py`'s `BLANK_TOKEN`). Loading any of these files with the current
   `build_model()` will fail with a shape mismatch. **Delete `models/*.h5` and retrain from
   scratch.**

2. **`data/` contains MIRACL-VC1-derived clips with placeholder labels, not real GRID
   transcripts.** The `.align` files in `data/alignments/S1/` contain entries like
   `0.5 1.5 03` - numeric placeholders, not the word-level transcripts the GRID corpus (and this
   model's vocabulary/architecture) are designed around. Training on this data will run without
   crashing (the CTC pipeline and shapes are correct - that's what the test suite verifies), but
   the model will only ever learn to predict these placeholder tokens, not real speech.
   See [DATA_SOURCES.md](DATA_SOURCES.md) for where to get the real GRID corpus.

## To get a working model

1. Download real GRID corpus videos + alignments (see [DATA_SOURCES.md](DATA_SOURCES.md)) and
   place them under `data/<speaker>/` with matching `data/alignments/<speaker>/*.align` files.
2. Delete the stale checkpoints: `rm models/*.h5`.
3. Train from scratch: `python3 -m src.train --epochs 50` (or however many epochs are needed;
   see `src/config.py` for `EPOCHS`, `LR_DECAY_START_EPOCH`, etc.).
4. Predict: `python3 -m src.predict <video> --weights models/weights_epoch_NN.h5`.

No training was performed as part of this status update - this document only describes what
exists in the repo and what's needed to get a real model trained.
