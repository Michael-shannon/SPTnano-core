# SPTnano-core

Single-particle tracking analysis, from raw `.nd2` movies to behaviour states predicted by a raw-motion transformer.
This is the cleaned-up version of the pipeline used for the paper. It has one notebook per step, the trained model,
and the settings that were used.

## Install

Windows (all notebooks):

```bash
conda env create -f environment.yml
conda activate sptnano
pip install -e .
```

WSL with a GPU (only needed to retrain, step 5):

```bash
conda env create -f environment_wsl.yml
```

Then tell the code where your data is:

```bash
copy configs\paths.example.yaml configs\paths.local.yaml     # then edit the folders in it
```

`paths.local.yaml` is not committed. Every notebook and script reads it, plus `configs/pipeline.yaml` (all settings)
and `configs/gates.yaml` (the gate geometry).

## The pipeline

| Step | Notebook | What it does | Runs on |
|---|---|---|---|
| 1 | `01_raw_images_to_tracks` | rename files, crop 150 × 150 px, MATLAB tracking ([docs/MATLAB_TRACKING.md](docs/MATLAB_TRACKING.md)), load and clean tracks | Windows + MATLAB |
| 2 | `02_window_features_and_master_tables` | 60-frame windows with ~30 motion features each, then merge all experiments into the master tables | cluster ([hpc/README.md](hpc/README.md)) or Windows |
| 3 | `03_gating` | label windows bound / confined / exploratory / transport with two gates, plus the stricter "certain" labels | Windows |
| 4 | `04_prepare_transformer` | low-train / high-test splits, tokens, augmentation and the augmentation screen | Windows |
| 5 | `05_train_transformer` | contrastive pretraining, then the classifier | WSL GPU (`wsl/train_transformer.sh`) |
| 6 | `06_evaluate_and_predict` | held-out predictions, scores, confusion matrix, baselines, new data | Windows |
| 7 | `07_paper_figures` | Figure 1 and 2 panels | Windows |
| 8 | `08_plasticity` | superwindows: state sequences, switching, dwell times, galleries, movies | Windows |
| 9 | `09_contours_and_movies` | cell contours, raw-movie overlays, live state + softmax movies | Windows (napari) |

Each notebook can be started on its own: it loads what the previous steps saved. The trained model is in
`models/block3_long_large/`, so steps 6–9 work without retraining.

## Scripts

Same code as the notebooks, for long runs from a terminal:

```bash
python scripts/compute_windows.py --input_file <experiment>_analyze/saved_data/instant_df.csv --n_jobs 16
python scripts/make_splits.py --out_dir <new splits folder>
python scripts/train.py --smoke_test             # use wsl/train_transformer.sh for the real run
python scripts/run_benchmarks.py
python scripts/predict.py --instant <folder>/instant_df_processed.csv --windowed <folder>/time_windowed_df_processed.csv
```

## Notes
- Notebooks that save tables (02, 03, 04, 06) write to new dated folders, so the data in use is never overwritten.
  Point `paths.local.yaml` at the new folder when you want to use it.
- The figure notebooks (07–09) save into `plots_dir` and overwrite figures with the same name.
- `make_splits` with the settings in `pipeline.yaml` gives the same train / val / test sizes as the paper split
  in `splits_dir`. Use the paper split itself when you want to compare with the paper numbers.

See [MODEL_CARD.md](MODEL_CARD.md) for the model being used, its training data and its scores.
