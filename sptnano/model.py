"""The transformer: block3 tokens -> 16-D window embedding -> 4 behaviour classes.

Encoder: Linear(3 -> 16), a learned CLS token in front, 3 standard TransformerEncoder layers
(4 heads, feed-forward 256, dropout 0.1), LayerNorm on the CLS output = the embedding.
Classifier head: Linear(16, 32) - ReLU - Dropout(0.2) - Linear(32, 4).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .config import fix_path

DEFAULT_ARCHITECTURE = {"input_dim": 3, "embed_dim": 16, "num_heads": 4, "ff_dim": 256, "num_layers": 3}


class TransformerMotionEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=16, num_heads=4, ff_dim=256, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, padding_mask=None):
        """tokens [B, T, 3], padding_mask [B, T+1] (True = ignore). Returns [B, embed_dim]."""
        batch_size = tokens.shape[0]
        x = self.input_proj(tokens)
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        out = self.transformer_encoder(x.permute(1, 0, 2), src_key_padding_mask=padding_mask)
        return self.norm(out[0])


class MotionClassifier(nn.Module):
    def __init__(self, encoder, num_classes=4, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.input_proj.out_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens, padding_mask=None, return_embedding=False):
        embedding = self.encoder(tokens, padding_mask)
        logits = self.head(embedding)
        return (logits, embedding) if return_embedding else logits


def build_encoder(architecture=None):
    a = {**DEFAULT_ARCHITECTURE, **(architecture or {})}
    return TransformerMotionEncoder(int(a["input_dim"]), int(a["embed_dim"]), int(a["num_heads"]), int(a["ff_dim"]), int(a["num_layers"]))


def read_manifest(model_dir):
    """run_manifest.json -> first run (the folder holds one trained model)."""
    with open(Path(model_dir) / "run_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)[0]


def load_classifier(model_dir, device="cpu"):
    """Load models/<run>/classifier/best_model.pt. Returns (model, class_labels)."""
    model_dir = fix_path(model_dir)
    run = read_manifest(model_dir)
    checkpoint_path = fix_path(run["model_dir"]) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", {})
    idx_to_label = {int(v): k for k, v in checkpoint["str_to_idx"].items()}
    class_labels = [idx_to_label[i] for i in sorted(idx_to_label)]
    model = MotionClassifier(
        build_encoder(architecture),
        num_classes=len(class_labels),
        hidden_dim=int(architecture.get("mlp_hidden_dim", 32)),
        dropout=float(architecture.get("mlp_dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {checkpoint_path} (best epoch {checkpoint.get('best_epoch', -1) + 1}), classes {class_labels}")
    return model, class_labels
