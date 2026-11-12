# Model card: block3_long_large

The raw-motion transformer used in the paper. Files: `models/block3_long_large/`.

## What it does

Input: one 60-frame window of a single-molecule track (10 ms per frame, positions in µm).
Output: probabilities for four motion states (bound, confined, exploratory, transport) and a 16-D embedding.
The softmax is then read as a behaviour state (`softmax_states:` in `configs/pipeline.yaml`):

- top-1 ≥ 0.70 → that state
- top-1 ≤ 0.70, top-2 ≥ 0.20 and top-1 − top-2 ≤ 0.40 → a boundary pair, e.g. `bound_confined`
- otherwise → `unresolved_<top-1>`

## Input tokens

Each window becomes 20 tokens of 3 frames: (sum dx, sum dy, circular-mean heading). The model sees only raw motion,
none of the window features that the gates were drawn on.

## Architecture

Transformer encoder: input 3 → embedding 16, 4 heads, feed-forward 256, 3 layers, CLS token.
Classifier: MLP head (32 hidden units, dropout 0.2) on the CLS embedding.

## Training

1. Contrastive pretraining (InfoNCE, temperature 0.2, batch 1024, up to 400 epochs; best epoch 243).
   Positive pair = a window and an augmented copy (segments of 10 tokens shuffled, rescaled 0.5–1.5×,
   rotated 1–10°). Windows from the same track are not used as negatives. Augmentation was chosen with the
   screen in notebook 04.
2. Classifier fine-tuning (cross-entropy, batch 512, up to 200 epochs; best epoch 99), on a class-balanced 10%
   of the labelled training windows.

Labels: `final_population_certain` from the hierarchical gates (notebook 03), without `unknown` and `ungatedgate1`.
Split ("low train / high test"): every R3 track and the 12 cells with the most tracks in each condition are test;
the rest is 90/10 train/val by whole track. 82,765 labelled training windows.

## Scores

Held-out test set, 590,814 windows with a certain label:

| Method | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| **Transformer (block3_long_large)** | **0.927** | **0.813** | **0.750** |
| Random forest on block3 tokens (same input as the transformer) | 0.933 | 0.619 | 0.670 |
| Gaussian HMM on per-frame motion (4 states, unsupervised) | 0.903 | 0.485 | 0.447 |
| Random forest on window features, without the 4 gate features | 0.994 | 0.968 | 0.969 |
| Random forest on all window features | 1.000 | 1.000 | 1.000 |

How to read this:

- The labels are drawn on four window features (speed, self-intersections, anomalous exponent, cumulative displacement).
  A random forest given those features just redraws the gates, so the two window-feature forests are upper bounds,
  not competitors. Dropping the four gate features is not enough, because correlated features (step length, MSD) are still there.
- The fair comparison is the random forest on the same block3 tokens: the transformer is much better on the rare
  classes (balanced accuracy 0.81 vs 0.62).
- Accuracy is dominated by exploratory windows (72% of the test set). Use balanced accuracy and macro F1.

Per-class (transformer): bound F1 0.88, confined 0.54, exploratory 0.98, transport 0.59.
Confined and transport have low precision: many windows the gates call exploratory or bound get these labels,
usually as boundary pairs (`confined_exploratory`, `bound_confined`).

Reproduce: notebook 06, and `python scripts/run_benchmarks.py`.

## Limits

- Trained on 10 ms / 65 nm data. Other frame rates or pixel sizes change step lengths and need retraining.
- The labels come from gates, so the model learns the gates' definition of each state, smoothed by its own
  view of the raw motion.
- Windows shorter than 60 frames are padded; they were not part of training.
