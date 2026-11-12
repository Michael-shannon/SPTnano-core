"""Turn each window into the transformer input.

Per frame: dx, dy (um, first frame 0) and direction_rad. Windows are padded/truncated to 60 frames.
block3: every 3 frames become one token = (sum dx, sum dy, circular mean of direction),
so a 60-frame window is 20 tokens, followed by zero padding up to 60 slots.

Padding mask has length 61 (slot 0 is the CLS token); True = padding, ignored by attention.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

MAX_SEQ_LEN = 60
BLOCK_SIZE = 3
FEATURE_COLUMNS = {"frame", "x_um", "y_um", "direction_rad"}


def frame_features(x_um, y_um, direction_rad=None):
    """(n, 3) float32 array of dx, dy, direction."""
    dx = np.diff(x_um, prepend=x_um[0])
    dy = np.diff(y_um, prepend=y_um[0])
    if direction_rad is None:
        direction_rad = np.nan_to_num(np.arctan2(dy, dx), nan=0.0)
    features = np.column_stack([dx, dy, direction_rad])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pad_or_truncate(features, length=MAX_SEQ_LEN):
    n, d = features.shape
    if n >= length:
        return features[:length].copy()
    return np.vstack([features, np.zeros((length - n, d), dtype=np.float32)])


def padding_mask(n_valid, length=MAX_SEQ_LEN):
    """[1 + length] bool; True for padded slots. Slot 0 is the CLS token."""
    n = int(max(0, min(n_valid, length)))
    mask = torch.zeros(1 + length, dtype=torch.bool)
    mask[1 + n:] = True
    return mask


def circular_mean(angles):
    return np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())


def block3_tokens(features, n_valid, block_size=BLOCK_SIZE):
    """Group the valid frames into blocks of 3. Returns (features padded to the same length, number of tokens)."""
    n_valid = int(max(0, min(int(n_valid), features.shape[0])))
    out = np.zeros_like(features[:, :3], dtype=np.float32)
    if n_valid == 0:
        return out, 0
    rows = []
    for start in range(0, n_valid, block_size):
        block = features[start:min(n_valid, start + block_size), :3]
        rows.append([block[:, 0].sum(), block[:, 1].sum(), circular_mean(block[:, 2])])
    out[:len(rows)] = np.asarray(rows, dtype=np.float32)
    return out, len(rows)


class WindowDataset(Dataset):
    """One sample per window_uid, already in block3 tokens.

    Without label_lookup: every window (contrastive pretraining, prediction).
    With label_lookup {window_uid: label} and str_to_idx: only windows with a known label (classifier training).
    keep_metadata: also keep the other columns of the first row (used for prediction output).
    """

    def __init__(self, instant_df, label_lookup=None, str_to_idx=None, keep_metadata=False, max_seq_len=MAX_SEQ_LEN, show_progress=True):
        df = instant_df if isinstance(instant_df, pl.DataFrame) else pl.from_pandas(instant_df)
        if label_lookup is not None:
            df = df.join(pl.DataFrame({"window_uid": list(label_lookup)}), on="window_uid", how="semi")
        df = df.sort(["window_uid", "frame"])
        has_direction = "direction_rad" in df.columns
        self.max_seq_len = max_seq_len
        self.metadata_cols = [c for c in df.columns if c not in FEATURE_COLUMNS] if keep_metadata else []
        self.samples = []

        windows = df.partition_by("window_uid", maintain_order=True)
        for sub in tqdm(windows, desc="windows", disable=not show_progress):
            window_uid = sub["window_uid"][0]
            sample = {"window_uid": str(window_uid), "unique_id": str(sub["unique_id"][0])}
            if label_lookup is not None:
                label = label_lookup.get(window_uid)
                if label not in str_to_idx:
                    continue
                sample["label"] = str_to_idx[label]
            features = frame_features(
                sub["x_um"].to_numpy(), sub["y_um"].to_numpy(),
                sub["direction_rad"].to_numpy() if has_direction else None,
            )
            n_frames = min(features.shape[0], max_seq_len)
            tokens, n_tokens = block3_tokens(pad_or_truncate(features, max_seq_len), n_frames)
            sample["tokens"] = tokens
            sample["n_tokens"] = n_tokens
            for col in self.metadata_cols:
                sample.setdefault(col, str(sub[col][0]))
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def labels(self):
        return np.array([s["label"] for s in self.samples], dtype=np.int64)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "tokens": torch.from_numpy(s["tokens"]),
            "padding_mask": padding_mask(s["n_tokens"], self.max_seq_len),
            "n_tokens": torch.tensor(s["n_tokens"], dtype=torch.long),
            "window_uid": s["window_uid"],
            "unique_id": s["unique_id"],
        }
        if "label" in s:
            item["label"] = torch.tensor(s["label"], dtype=torch.long)
        for col in self.metadata_cols:
            item.setdefault(col, s[col])
        return item


def load_label_lookup(windowed_parquet, label_col="final_population_certain", exclude_labels=("unknown", "ungatedgate1")):
    """{window_uid: label} from the gated window table, without excluded labels."""
    sub = (
        pl.read_parquet(windowed_parquet, columns=["window_uid", label_col])
        .drop_nulls(subset=[label_col])
        .unique(subset=["window_uid"], keep="last")
        .filter(~pl.col(label_col).cast(pl.Utf8).is_in(list(exclude_labels)))
    )
    return dict(zip(sub["window_uid"].cast(pl.Utf8).to_list(), sub[label_col].cast(pl.Utf8).to_list()))


def label_index(label_lookup):
    """Classes in alphabetical order: {bound: 0, confined: 1, exploratory: 2, transport: 3}."""
    return {label: i for i, label in enumerate(sorted(set(label_lookup.values())))}
