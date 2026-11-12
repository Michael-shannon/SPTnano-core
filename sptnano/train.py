"""Step 5: train the transformer in two stages.

Stage 1  contrastive pretraining (InfoNCE) of the encoder on all train windows:
         clean window vs its augmented copy; other windows from the same track are not used as negatives.
Stage 2  classifier: pretrained encoder + small MLP head, cross-entropy on a balanced 10% subset of the
         labelled train windows (2,070 per class for the paper model), stratified sampler,
         loss = mean of CE(clean) and CE(augmented).

Both stages: Adam lr 1e-4, ReduceLROnPlateau(0.5, patience 5) on val loss, keep the best val epoch,
early stop after 15 epochs without improvement, checkpoint every 5 epochs.
Run on the GPU in WSL: wsl/train_transformer.sh.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augment import make_augmentation, zero_padding
from .model import MotionClassifier, build_encoder
from .tokens import WindowDataset, label_index, load_label_lookup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def masked_infonce_loss(z_clean, z_aug, unique_ids, temperature=0.2):
    """Each window's positive is its own augmented copy. Windows from the same track are removed from the negatives."""
    z_clean = F.normalize(z_clean, dim=1)
    z_aug = F.normalize(z_aug, dim=1)
    reps = torch.cat([z_clean, z_aug], dim=0)
    sim = torch.matmul(reps, reps.T) / float(temperature)
    n = z_clean.shape[0]
    device = z_clean.device

    positive_idx = torch.cat([torch.arange(n, device=device) + n, torch.arange(n, device=device)])
    _, track_codes = np.unique(np.asarray([str(u) for u in unique_ids]), return_inverse=True)
    track_codes = torch.as_tensor(np.concatenate([track_codes, track_codes]), device=device)
    same_track = track_codes[:, None] == track_codes[None, :]
    is_positive = torch.zeros((2 * n, 2 * n), dtype=torch.bool, device=device)
    is_positive[torch.arange(2 * n, device=device), positive_idx] = True
    keep = ~torch.eye(2 * n, dtype=torch.bool, device=device) & (~same_track | is_positive)

    positives = sim[torch.arange(2 * n, device=device), positive_idx]
    return (-positives + torch.logsumexp(sim.masked_fill(~keep, float("-inf")), dim=1)).mean()


def make_loaders(train_ds, val_ds, batch_size, workers, sampler=None):
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def balanced_label_subset(ds, fraction=0.10, seed=42, min_per_class=100):
    """Copy of ds with the same number of windows from every class.

    per class = max(ceil(fraction * n_windows / n_classes), min_per_class), capped at what the class has.
    """
    labels = ds.labels()
    classes = np.unique(labels)
    total_target = max(len(classes), int(np.ceil(labels.size * fraction)))
    per_class = max(int(np.ceil(total_target / len(classes))), int(min_per_class))
    rng = np.random.default_rng(seed)
    keep = []
    counts_before, counts_after = {}, {}
    for cls in classes:
        idx = np.where(labels == cls)[0]
        n_keep = min(idx.size, per_class)
        keep.extend(rng.choice(idx, size=n_keep, replace=False).tolist())
        counts_before[int(cls)], counts_after[int(cls)] = int(idx.size), int(n_keep)
    subset = copy.copy(ds)
    subset.samples = [ds.samples[i] for i in sorted(keep)]
    info = {"enabled": True, "fraction_requested": fraction, "n_before": int(labels.size), "n_after": len(subset.samples),
            "classes": [int(c) for c in classes], "per_class_target": per_class,
            "counts_before": counts_before, "counts_after": counts_after}
    return subset, info


def stratified_sampler(labels, n_classes):
    """Every class is drawn equally often (weights 1 / class count, with replacement)."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / np.maximum(counts[labels], 1)
    return WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples=len(labels), replacement=True)


class Trainer:
    """Shared loop: best val epoch, LR scheduler, checkpoints, early stopping. Subclasses define step()."""

    def __init__(self, model, train_loader, val_loader, augment, device, t):
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.augment = augment
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=t["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=t["scheduler_factor"], patience=t["scheduler_patience"], min_lr=0.0
        )
        self.early_stopping_patience = t["early_stopping_patience"]
        self.checkpoint_interval = t["checkpoint_interval"]
        self.train_losses, self.val_losses, self.val_accuracies = [], [], []
        self.best_val_loss, self.best_epoch, self.best_model_state = float("inf"), -1, None

    def batch_to_device(self, batch):
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def augmented(self, batch):
        return zero_padding(self.augment(batch["tokens"], batch["n_tokens"]), batch["padding_mask"])

    def train_epoch(self):
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.train_step(self.batch_to_device(batch))
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
        return total / max(len(self.train_loader), 1)

    def fit(self, epochs, out_dir):
        checkpoint_dir = Path(out_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        epochs_without_improvement = 0
        for epoch in range(int(epochs)):
            train_loss = self.train_epoch()
            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            marker = ""
            if val_loss < self.best_val_loss:
                self.best_val_loss, self.best_epoch = val_loss, epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                marker = " best"
            else:
                epochs_without_improvement += 1
            self.scheduler.step(val_loss)
            if self.checkpoint_interval > 0 and (epoch + 1) % self.checkpoint_interval == 0:
                torch.save(self.payload({"epoch": epoch}), checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt")
            acc_text = f" val_acc={val_acc:.4f}" if val_acc is not None else ""
            print(f"epoch {epoch + 1:3d}  train={train_loss:.4f}  val={val_loss:.4f}{acc_text}  lr={self.optimizer.param_groups[0]['lr']:.1e}{marker}")
            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"early stop: no val improvement for {self.early_stopping_patience} epochs")
                break

    def payload(self, extra=None):
        out = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "best_model_state": self.best_model_state,
        }
        out.update(extra or {})
        return out

    def save(self, out_dir, config):
        """final_model.pt = last epoch, best_model.pt = best val epoch."""
        torch.save(self.payload(config), Path(out_dir) / "final_model.pt")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            torch.save(self.payload(config), Path(out_dir) / "best_model.pt")
        print(f"saved {out_dir} (best epoch {self.best_epoch + 1}, val loss {self.best_val_loss:.4f})")


class ContrastiveTrainer(Trainer):
    def __init__(self, *args, temperature=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def loss(self, batch):
        z_clean = self.model(batch["tokens"], batch["padding_mask"])
        z_aug = self.model(self.augmented(batch), batch["padding_mask"])
        return masked_infonce_loss(z_clean, z_aug, batch["unique_id"], self.temperature)

    def train_step(self, batch):
        return self.loss(batch)

    def validate(self):
        losses = [float(self.loss(self.batch_to_device(b)).item()) for b in self.val_loader]
        return sum(losses) / max(len(losses), 1), None


class ClassifierTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce = nn.CrossEntropyLoss()

    def train_step(self, batch):
        logits = self.model(batch["tokens"], batch["padding_mask"])
        logits_aug = self.model(self.augmented(batch), batch["padding_mask"])
        return 0.5 * (self.ce(logits, batch["label"]) + self.ce(logits_aug, batch["label"]))

    def validate(self):
        total, correct, count = 0.0, 0, 0
        for batch in self.val_loader:
            batch = self.batch_to_device(batch)
            logits = self.model(batch["tokens"], batch["padding_mask"])
            total += float(self.ce(logits, batch["label"]).item())
            correct += int((logits.argmax(dim=1) == batch["label"]).sum().item())
            count += int(batch["label"].numel())
        return total / max(len(self.val_loader), 1), correct / max(count, 1)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def train_block3_model(splits_dir, windowed_parquet, out_dir, settings, smoke_test=False, device=None):
    """Both stages end to end. Writes out_dir/{contrastive_pretrain, classifier}/ and out_dir/run_manifest.json."""
    t = settings["transformer"]
    s = settings["splits"]
    out_dir = Path(out_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(t["seed"])

    train_df = pl.read_parquet(Path(splits_dir) / "train_df.parquet")
    val_df = pl.read_parquet(Path(splits_dir) / "val_df.parquet")
    contrastive_epochs, finetune_epochs = t["contrastive"]["epochs"], t["finetune"]["epochs"]
    contrastive_batch, finetune_batch, workers = t["contrastive"]["batch_size"], t["finetune"]["batch_size"], t["num_workers"]
    if smoke_test:
        keep_train = train_df["window_uid"].unique().sort().head(2048)
        keep_val = val_df["window_uid"].unique().sort().head(512)
        train_df = train_df.filter(pl.col("window_uid").is_in(keep_train))
        val_df = val_df.filter(pl.col("window_uid").is_in(keep_val))
        contrastive_epochs = finetune_epochs = 1
        contrastive_batch, finetune_batch, workers = 128, 64, 0
        out_dir = out_dir.parent / f"{out_dir.name}_smoke"
    print(f"train rows {train_df.height:,}  val rows {val_df.height:,}  device {device}")

    label_lookup = load_label_lookup(windowed_parquet, s["label_col"], s["exclude_labels"])
    str_to_idx = label_index(label_lookup)
    print("classes:", str_to_idx)
    train_ds = WindowDataset(train_df, label_lookup, str_to_idx)
    val_ds = WindowDataset(val_df, label_lookup, str_to_idx)
    augment = make_augmentation(t["augmentation_segment_tokens"])
    architecture = {k: t[k] for k in ("embed_dim", "num_heads", "ff_dim", "num_layers")} | {"input_dim": 3}
    data_paths = {"splits_path": str(splits_dir), "windowed_df_path": str(windowed_parquet)}

    # Stage 1: contrastive pretraining
    pretrain_dir = out_dir / "contrastive_pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    pretrain_config = {
        "model_name": "block3_contrastive_infonce", "stage": "contrastive_pretrain", "feature_mode": "block3",
        "architecture": architecture, "temperature": t["contrastive"]["temperature"], "mask_same_unique_id": True,
        "augmentation_type": "shuffle_scale_angle", "shuffle_segment_length_tokens": t["augmentation_segment_tokens"],
        "epochs_target": contrastive_epochs, "batch_size": contrastive_batch, "learning_rate": t["learning_rate"], **data_paths,
    }
    write_json(pretrain_dir / "config.json", pretrain_config)
    train_loader, val_loader = make_loaders(train_ds, val_ds, contrastive_batch, workers)
    pretrainer = ContrastiveTrainer(build_encoder(architecture), train_loader, val_loader, augment, device, t,
                                    temperature=t["contrastive"]["temperature"])
    pretrainer.fit(contrastive_epochs, pretrain_dir)
    pretrainer.save(pretrain_dir, pretrain_config)
    encoder_state = copy.deepcopy(pretrainer.model.state_dict())
    pretrain_result = {"model_dir": str(pretrain_dir), "best_val_loss": pretrainer.best_val_loss, "best_epoch": pretrainer.best_epoch}
    del pretrainer, train_loader, val_loader

    # Stage 2: classifier on a balanced 10% label subset
    f = t["finetune"]
    classifier_dir = out_dir / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    subset_ds, subset_info = balanced_label_subset(train_ds, f["subset_fraction"], f["subset_seed"], f["subset_min_per_class"])
    print(f"label subset: {subset_info['n_before']:,} -> {subset_info['n_after']:,} windows, {subset_info['counts_after']}")
    encoder = build_encoder(architecture)
    encoder.load_state_dict(encoder_state)
    model = MotionClassifier(encoder, num_classes=len(str_to_idx), hidden_dim=t["mlp_hidden_dim"], dropout=t["mlp_dropout"])
    classifier_config = {
        "model_name": "block3_classifier_ce", "stage": "finetune", "feature_mode": "block3",
        "architecture": architecture | {"head_type": "mlp", "mlp_hidden_dim": t["mlp_hidden_dim"],
                                        "mlp_dropout": t["mlp_dropout"], "num_classes": len(str_to_idx)},
        "str_to_idx": str_to_idx, "idx_to_str": {v: k for k, v in str_to_idx.items()}, "num_classes": len(str_to_idx),
        "label_column": s["label_col"], "exclude_labels": sorted(s["exclude_labels"]), "supervised_subset": subset_info,
        "pretrained_encoder": pretrain_result,
        "epochs_target": finetune_epochs, "batch_size": finetune_batch, "learning_rate": t["learning_rate"], **data_paths,
    }
    write_json(classifier_dir / "config.json", classifier_config)
    sampler = stratified_sampler(subset_ds.labels(), len(str_to_idx))
    train_loader, val_loader = make_loaders(subset_ds, val_ds, finetune_batch, workers, sampler=sampler)
    classifier_trainer = ClassifierTrainer(model, train_loader, val_loader, augment, device, t)
    classifier_trainer.fit(finetune_epochs, classifier_dir)
    classifier_trainer.save(classifier_dir, classifier_config)

    manifest = [{
        "run_id": out_dir.name, "description": "Block-averaged 3-frame raw motion tokens.", "feature_mode": "block3",
        "architecture": architecture,
        "pretrain_model_dir": str(pretrain_dir), "pretrain_best_val_loss": pretrain_result["best_val_loss"],
        "pretrain_best_epoch": pretrain_result["best_epoch"],
        "model_dir": str(classifier_dir), "best_val_loss": classifier_trainer.best_val_loss, "best_epoch": classifier_trainer.best_epoch,
    }]
    write_json(out_dir / "run_manifest.json", manifest)
    print(f"done: {out_dir}")
    return out_dir
