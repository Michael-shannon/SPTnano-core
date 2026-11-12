"""Step 6b: how good is the model? Scores on held-out windows, confusion matrix, loss curves, embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from .tokens import load_label_lookup

CLASS_ORDER = ["bound", "confined", "exploratory", "transport"]


def attach_gate_labels(window_df, windowed_parquet, label_cols=("final_population", "final_population_certain")):
    """Add the gate labels (used only for scoring, never for the prediction itself)."""
    out = window_df.drop(columns=[c for c in label_cols if c in window_df.columns])
    for col in label_cols:
        lookup = load_label_lookup(windowed_parquet, col, exclude_labels=())
        out[col] = out["window_uid"].astype(str).map(lookup)
    return out


def classification_summary(df, true_col="final_population_certain", pred_col="pred_label", labels=CLASS_ORDER):
    """Accuracy, balanced accuracy (= macro recall) and macro F1 on windows that have one of the 4 labels."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

    sub = df[df[true_col].isin(labels)]
    y_true, y_pred = sub[true_col].astype(str), sub[pred_col].astype(str)
    scores = {
        "n_windows": len(sub),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro"),
    }
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()})
    report = pd.DataFrame(classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)).T
    return scores, report


def plot_confusion_matrix_report(
    df: pd.DataFrame,
    *,
    true_col: str,
    pred_col: str,
    output_dir: str | Path,
    name: str,
    labels: list[str] | None = None,
    normalize: str | None = "true",
    figsize: tuple[float, float] = (8, 7),
    export_format: str = "svg",
) -> pd.DataFrame:
    """Confusion matrix figure plus classification_report csv, saved as <name>_confusion_matrix / <name>_classification_report."""
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sub = df[[true_col, pred_col]].dropna().copy()
    sub[true_col] = sub[true_col].astype(str)
    sub[pred_col] = sub[pred_col].astype(str)
    if labels is None:
        labels = sorted(set(sub[true_col]).union(set(sub[pred_col])))

    cm = confusion_matrix(sub[true_col], sub[pred_col], labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d", colorbar=True)
    ax.set_title(f"{name}: {true_col} vs {pred_col}")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    plt.tight_layout()
    fig_path = output_dir / f"{name}_confusion_matrix.{export_format}"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved:", fig_path)

    report = classification_report(sub[true_col], sub[pred_col], labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_path = output_dir / f"{name}_classification_report.csv"
    report_df.to_csv(report_path)
    print("Saved:", report_path)
    return report_df


def loss_table(model_dir):
    """Train/val loss per epoch for both stages (both checkpoint files keep the full history)."""
    rows = []
    for stage, sub in (("contrastive pretraining", "contrastive_pretrain"), ("classifier", "classifier")):
        path = Path(model_dir) / sub / "final_model.pt"
        if not path.exists():
            path = path.with_name("best_model.pt")
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for epoch, (train, val) in enumerate(zip(checkpoint["train_losses"], checkpoint["val_losses"]), start=1):
            rows.append({"stage": stage, "epoch": epoch, "train": train, "validation": val})
    return pd.DataFrame(rows)


def plot_loss_curves(losses, save_path=None):
    stages = list(dict.fromkeys(losses["stage"]))
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 3.5), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        sub = losses[losses["stage"] == stage]
        ax.plot(sub["epoch"], sub["train"], label="train", color="0.3")
        ax.plot(sub["epoch"], sub["validation"], label="validation", color="#d95f02")
        ax.set_title(stage)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", transparent=True)
    return fig


def compute_embedding_2d(
    df: pd.DataFrame,
    *,
    emb_cols: list[str] | None = None,
    method: str = "umap",
    n_sample: int | None = 100_000,
    random_seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """UMAP or PHATE of the 16-D embedding columns (emb_00..emb_15). Adds umap_1/umap_2 or phate_1/phate_2."""
    if emb_cols is None:
        emb_cols = [col for col in df.columns if col.startswith("emb_")]
    if not emb_cols:
        raise KeyError("No embedding columns found. Expected columns like emb_00, emb_01, ...")

    plot_df = df.copy()
    if n_sample is not None and len(plot_df) > n_sample:
        plot_df = plot_df.sample(n_sample, random_state=random_seed)

    x = plot_df[emb_cols].to_numpy(dtype=float)
    method = method.lower()
    if method == "umap":
        import umap

        reducer = umap.UMAP(
            n_neighbors=int(kwargs.get("n_neighbors", 30)),
            min_dist=float(kwargs.get("min_dist", 0.1)),
            metric=kwargs.get("metric", "euclidean"),
            random_state=random_seed,
        )
        coords = reducer.fit_transform(x)
        prefix = "umap"
    elif method == "phate":
        import phate

        reducer = phate.PHATE(
            n_components=2,
            knn=int(kwargs.get("knn", 15)),
            decay=int(kwargs.get("decay", 40)),
            t=kwargs.get("t", "auto"),
            random_state=random_seed,
            verbose=0,
        )
        coords = reducer.fit_transform(x)
        prefix = "phate"
    else:
        raise ValueError("method must be 'umap' or 'phate'")

    out = plot_df.copy()
    out[f"{prefix}_1"] = coords[:, 0]
    out[f"{prefix}_2"] = coords[:, 1]
    return out


def plot_embedding(embedding_df, x_col, y_col, color_col, palette=None, save_path=None, point_size=2, alpha=0.35, figsize=(6, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    order = [c for c in (palette or {}) if c in set(embedding_df[color_col])] or None
    sns.scatterplot(data=embedding_df, x=x_col, y=y_col, hue=color_col, hue_order=order, palette=palette,
                    s=point_size, alpha=alpha, linewidth=0, rasterized=True, ax=ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title=color_col, markerscale=4, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches="tight", transparent=True)
    return fig, ax


def state_counts(df, col="transformer_behavior_state"):
    counts = df[col].value_counts().rename("n_windows").to_frame()
    counts["fraction"] = counts["n_windows"] / counts["n_windows"].sum()
    return counts


def mean_softmax_by_label(df, true_col="final_population_certain", labels=CLASS_ORDER):
    """Average predicted probability of each state, per gate label (a softer confusion matrix)."""
    sub = df[df[true_col].isin(labels)]
    return sub.groupby(true_col)[[f"prob_{c}" for c in labels]].mean().reindex(labels)


def summary_line(scores):
    return (f"{scores['n_windows']:,} held-out windows: accuracy {scores['accuracy']:.3f}, "
            f"balanced accuracy {scores['balanced_accuracy']:.3f}, macro F1 {scores['macro_f1']:.3f}")
