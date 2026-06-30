# Project Status

## Code: implemented and tested

- **Data pipeline** (`src/data.py`, `src/dataset.py`): video loading, mouth-region cropping,
  alignment parsing (handles both `sil` and `silence` tokens), `tf.data` batching/padding.
- **Model** (`src/model.py`): Conv3D + BiLSTM + CTC, ~12M parameters.
- **Training** (`src/train.py`, `src/losses.py`, `src/callbacks.py`): CTC loss, learning-rate
  schedule, checkpointing, example-prediction callback.
- **Prediction** (`src/predict.py`): loads a checkpoint and decodes a video to text, with greedy
  or beam-search CTC decoding (`--decoding greedy|beam`).
- **Evaluation** (`src/metrics.py`, `eval_sample.py`): word error rate (WER) and character error
  rate (CER) against real `.align` transcripts.
- **Visualization** (`src/visualize.py`): saves a preprocessed clip as a GIF.

All of the above is covered by the pytest suite under `tests/` (64 tests, 93% coverage of `src/`,
run in CI via `.github/workflows/ci.yml`). The suite uses small synthetic videos/alignments
generated on the fly, so it runs fully offline regardless of what real data is present locally.

## Real data and a real trained model exist - but only on the machine that made them

A model has actually been trained on the real GRID corpus, not placeholder data:

1. **Real GRID speaker `s1` data was downloaded from Zenodo** with `fetch_grid.py` (~13.5GB:
   `alignments.zip` + speaker zips `s1`-`s34`, minus `s21` which is still missing from the
   download). Only `s1` has been extracted into `data/s1/` + `data/alignments/s1/` (1001 clips);
   the other speaker zips remain unextracted in `downloads/`.
2. **A model was trained for 50 epochs on that `s1` data** (CPU, June 2026). Checkpoints
   `models/weights_epoch_01.h5` through `weights_epoch_50.h5` exist and load cleanly against the
   current `build_model()` (correct `vocab_size + 2` / CTC-blank-token shape - these are not the
   old incompatible checkpoints that earlier versions of this doc warned about).
3. **The best checkpoint by validation loss is epoch 41** (val_loss ~27.24); later epochs show
   mild overfitting on the single-speaker split (train loss keeps falling to ~11.75 by epoch 50
   while val_loss plateaus around 27-28). The example predictions in this README's
   "Demo / Example" section come from running `eval_sample.py` against epoch 41.
4. **Real WER/CER, computed over all 1000 `s1` clips against epoch 41:**
   - Greedy decoding: **WER 0.565, CER 0.286**
   - Beam search decoding: **WER 0.554, CER 0.262**

   Beam search wins on both metrics, consistent with the expectation that exploring multiple
   candidate paths helps on the short/ambiguous tokens (single letters, digits) that are the
   model's weakest class. These numbers are from a single-speaker (`s1`) model evaluated on a
   sample that includes clips used in training/validation (the train/val split's shuffle wasn't
   seeded, so it can't be reproduced from outside the training run) - treat them as representative
   of in-distribution performance on `s1`, not as a clean held-out/generalization number.

**None of this is in git.** `data/`, `downloads/`, and `models/*.h5` are gitignored (~14GB+
combined), so a fresh clone has no data and no trained model until you regenerate them yourself.

## To reproduce a working model from a fresh clone

1. Download real GRID corpus videos + alignments: `python fetch_grid.py` (resumable; see
   [DATA_SOURCES.md](DATA_SOURCES.md) for the manual Zenodo steps if you prefer). Extract so
   videos land under `data/<speaker>/` and alignments under `data/alignments/<speaker>/`.
2. Train: `python -m src.train --video_pattern "data/s1/*.mpg" --epochs 50` (or
   `data/s*/*.mpg` for multiple speakers - speaker-independent training hasn't been tried yet).
3. Evaluate: `python eval_sample.py --weights models/weights_epoch_NN.h5` for WER/CER on a
   sample of clips, or `python -m src.predict <video> --weights models/weights_epoch_NN.h5` for
   a single clip (add `--decoding beam` to try beam search instead of greedy).

## Known gaps

- Only one GRID speaker (`s1`) has been used for training; speaker-independent results across
  multiple speakers are untested.
- The WER/CER numbers above are not a clean held-out evaluation (see note above) - a proper
  train/val/test split with a fixed seed would be needed for a generalization number.
- `s21.zip` is missing from the Zenodo download and hasn't been re-fetched.
