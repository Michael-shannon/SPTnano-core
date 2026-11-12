"""The augmentation used for both training stages ("shuffle_scale_angle", valid prefix only).

For each window (on the block3 tokens):
  1. cut the valid tokens into segments of 10,
  2. scale each segment by 0.5-0.9 or 1.1-1.5 (coin flip),
  3. shuffle the segment order,
  4. rotate every step by 1-10 degrees, random sign.
Padding is left as zero. The seed comes from the batch position and the first 5 positions,
so a given window always gets the same augmentation at the same batch position.
"""

from __future__ import annotations

import numpy as np
import torch

SCALE_RANGE = (0.5, 1.5)
ANGLE_RANGE_DEG = (1, 10)


def shuffle_scale_rotate(x, y, segment_length, seed, scale_range=SCALE_RANGE, angle_range_deg=ANGLE_RANGE_DEG):
    """x, y: cumulative positions. Returns augmented cumulative positions."""
    np.random.seed(seed)
    n = len(x)
    if n == 0:
        return np.asarray(x, dtype=np.float64).copy(), np.asarray(y, dtype=np.float64).copy()
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    segment_length = max(1, int(segment_length))
    segments = []
    for start in range(0, n, segment_length):
        end = min(start + segment_length, n)
        if np.random.random() < 0.5:
            scale = np.random.uniform(scale_range[0], 0.9)
        else:
            scale = np.random.uniform(1.1, scale_range[1])
        segments.append((dx[start:end] * scale, dy[start:end] * scale))

    order = np.arange(len(segments))
    np.random.shuffle(order)
    new_dx = np.concatenate([segments[i][0] for i in order])
    new_dy = np.concatenate([segments[i][1] for i in order])

    min_rad, max_rad = np.deg2rad(angle_range_deg[0]), np.deg2rad(angle_range_deg[1])
    r = np.sqrt(new_dx**2 + new_dy**2)
    theta = np.arctan2(new_dy, new_dx)
    theta = theta + np.random.uniform(min_rad, max_rad, size=len(theta)) * np.random.choice([-1, 1], size=len(theta))
    return np.cumsum(r * np.cos(theta)), np.cumsum(r * np.sin(theta))


def augment_valid_prefix(x, y, n_valid, segment_length, seed):
    """Augment the first n_valid positions; the tail repeats the last position (zero steps)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_valid = int(max(0, min(n_valid, len(x))))
    if n_valid == 0:
        return x.astype(np.float32), y.astype(np.float32)
    xa, ya = shuffle_scale_rotate(x[:n_valid].copy(), y[:n_valid].copy(), segment_length, seed)
    out_x = np.full(len(x), xa[-1])
    out_y = np.full(len(y), ya[-1])
    out_x[:n_valid] = xa
    out_y[:n_valid] = ya
    return out_x.astype(np.float32), out_y.astype(np.float32)


def make_augmentation(segment_length=10):
    """Returns augment(tokens [B, T, 3], n_tokens [B]) -> augmented tokens."""

    def augment(tokens, n_tokens):
        device = tokens.device
        batch = tokens.detach().cpu().numpy()
        n_valid = n_tokens.detach().cpu().numpy()
        out = []
        for b in range(batch.shape[0]):
            x = np.cumsum(batch[b, :, 0])
            y = np.cumsum(batch[b, :, 1])
            seed = hash((b, tuple(x[:5]), tuple(y[:5]))) % 2**32
            xa, ya = augment_valid_prefix(x, y, int(n_valid[b]), segment_length, seed)
            dx = np.diff(xa, prepend=xa[0])
            dy = np.diff(ya, prepend=ya[0])
            direction = np.nan_to_num(np.arctan2(dy, dx), nan=0.0)
            out.append(np.nan_to_num(np.column_stack([dx, dy, direction]), nan=0.0, posinf=0.0, neginf=0.0))
        return torch.from_numpy(np.array(out)).float().to(device)

    return augment


def zero_padding(tokens, padding_mask):
    """Set padded slots back to 0 after augmentation (padding_mask includes the CLS slot)."""
    return tokens.masked_fill(padding_mask[:, 1:].unsqueeze(-1), 0.0)
