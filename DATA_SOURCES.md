# Data Sources for Lip-Reading Project

## GRID Corpus - Available on Zenodo!

**Good news!** The GRID corpus is available on Zenodo:

- **Zenodo URL**: https://zenodo.org/records/3625687
- **DOI**: 10.5281/zenodo.3625687
- **Size**: ~25GB (videos) + alignments

### How to Download:

1. Visit: https://zenodo.org/records/3625687
2. Download:
   - `alignments.zip` (required)
   - `s1.zip` through `s34.zip` (speaker videos - download as needed)
   - `audio_25k.zip` (optional - audio files)
3. Extract and organize using: `python3 download_grid_zenodo.py --extract downloads/`

## Alternative Datasets

If you prefer other datasets, here are alternatives:

### Option 1: Academic Repositories

1. **Kaggle**
   - Search for "GRID corpus" or "lip reading dataset"
   - Many researchers share processed versions
   - URL: https://www.kaggle.com/datasets

2. **Papers with Code**
   - Search for lip-reading papers that use GRID
   - Authors often provide dataset links
   - URL: https://paperswithcode.com/

3. **GitHub**
   - Search for "GRID dataset" or "lip reading GRID"
   - Many projects include download scripts
   - Example search: `site:github.com GRID corpus lip reading`

### Option 2: Alternative Datasets

If GRID is unavailable, you can use similar datasets:

1. **LRW (Lip Reading in the Wild)**
   - Larger dataset, more challenging
   - Available from: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/

2. **MIRACL-VC1**
   - Multimodal dataset
   - Check: https://sites.google.com/view/miracl-vc1

3. **LRS2/LRS3**
   - BBC dataset for lip reading
   - Requires research access

### Option 3: Create Your Own Dataset

You can create a small dataset for testing:

1. Record short videos of yourself speaking
2. Create alignment files manually
3. Use the same structure as GRID

### Option 4: Use Sample/Demo Data

For testing the code structure, you can:
1. Create a few sample videos (even synthetic)
2. Create corresponding alignment files
3. Test the pipeline with minimal data

## Required Data Structure

Regardless of source, your data should follow this structure:

```
data/
├── S1/              # Speaker 1 (or any speaker ID)
│   ├── video1.mpg
│   ├── video2.mpg
│   └── ...
└── alignments/
    └── S1/
        ├── video1.align
        ├── video2.align
        └── ...
```

### Alignment File Format

Each `.align` file should contain:
```
start_time end_time token
start_time end_time token
...
```

Example:
```
0 0.5 silence
0.5 1.0 hello
1.0 1.5 world
1.5 2.0 silence
```

## Quick Start with Minimal Data

If you just want to test the code, create a minimal dataset:

1. Create `data/S1/` directory
2. Add at least 2-3 video files (`.mpg` format)
3. Create `data/alignments/S1/` directory
4. Add corresponding `.align` files
5. Run training with a small number of epochs

This will help verify the code works before obtaining the full dataset.

