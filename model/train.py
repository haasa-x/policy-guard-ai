"""CHEM GAT training pipeline (Facebook Hateful Memes).

End-to-end training of :class:`model.gat_model.CHEMGraphAttentionNetwork` on the
Facebook Hateful Memes dataset.  Each meme (image + overlaid text + binary
label) is turned into a *Content Composition Graph* and the GAT learns to
classify the whole graph as SAFE / HARMFUL.

Pipeline
--------
1. **Load** Hateful Memes samples via ``data.preprocess.hateful_memes``.
2. **Encode** each sample with the CHEM text + image encoders (loaded once).
3. **Build** a PyG ``Data`` graph per sample via ``graph.graph_builder.build_graph``.
4. **Split** 70 / 15 / 15 into train / val / test and batch with PyG ``DataLoader``.
5. **Train** with ``NLLLoss`` on the model's ``log_softmax`` output (equivalent to
   BCE for the 2-class case), Adam (``lr=0.005``, ``weight_decay=5e-4``), with
   early stopping (patience 20) on validation F1; the best checkpoint is saved to
   ``model/checkpoints/best_gat.pt``.
6. **Log** every metric to Weights & Biases (guarded so training still runs if
   W&B is offline / unavailable).
7. **Evaluate** on the held-out test set, printing accuracy / precision / recall
   / F1 / AUC.

Expected data interface
------------------------
This module targets the documented interface
``data.preprocess.hateful_memes.load_hateful_memes(split) -> iterable of
{"id", "img", "text", "label"}``.  If the concrete module exposes a differently
named loader, adapt :func:`_load_samples` -- the rest of the pipeline is agnostic
to the loader's internals.

Run
---
``python -m model.train --epochs 100 --batch_size 32 --data_dir data/hateful_memes``
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from model.gat_model import CHEMGraphAttentionNetwork

# Default checkpoint location (created on demand).
_CHECKPOINT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "checkpoints"))
_BEST_CHECKPOINT_PATH = os.path.join(_CHECKPOINT_DIR, "best_gat.pt")

# Training hyper-parameters.
_LEARNING_RATE = 0.001        # lower LR works better with AdamW + scheduler
_WEIGHT_DECAY = 1e-3
_EARLY_STOP_PATIENCE = 30     # more patience for the scheduler to kick in
_LR_SCHEDULE_PATIENCE = 5     # halve LR if val F1 stalls for this many epochs


# --------------------------------------------------------------------------- #
# Data loading + graph construction
# --------------------------------------------------------------------------- #
def _load_samples(split: str, data_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Load raw Hateful Memes samples for a split via the preprocess module.

    Parameters
    ----------
    split : str
        Dataset split name (``"train"`` / ``"dev"`` / ``"test"`` etc.).
    data_dir : str, optional
        Optional dataset root forwarded to the loader if it accepts one.

    Returns
    -------
    list[dict]
        Samples shaped ``{"id", "img", "text", "label"}``.  Returns an empty
        list (with a warning) if the loader is unavailable, so callers can
        degrade gracefully.
    """
    try:
        from data.preprocess import hateful_memes
    except Exception as exc:  # pragma: no cover - preprocess may not be present
        print(f"[CHEM][train] WARNING: could not import hateful_memes loader: {exc}")
        return []

    # Prefer the documented `load_hateful_memes(split)` interface; fall back to
    # any single obvious loader callable the module exposes.
    loader = getattr(hateful_memes, "load_hateful_memes", None)
    if loader is None:
        for name in ("load", "load_split", "iter_samples"):
            loader = getattr(hateful_memes, name, None)
            if loader is not None:
                break
    if loader is None:
        print(
            "[CHEM][train] WARNING: hateful_memes exposes no known loader "
            "(expected load_hateful_memes(split))."
        )
        return []

    try:
        # load_hateful_memes(data_dir, split) — data_dir first, then split name.
        try:
            samples = loader(data_dir, split)  # type: ignore[misc]
        except TypeError:
            samples = loader(split)  # type: ignore[misc]
        return list(samples)
    except Exception as exc:  # pragma: no cover
        print(f"[CHEM][train] WARNING: loader failed for split={split!r}: {exc}")
        return []


def _build_dataset(
    samples: List[Dict[str, Any]],
    text_encoder,
    image_encoder,
) -> List[Any]:
    """Encode + graph-build every sample into a list of PyG ``Data`` objects.

    Parameters
    ----------
    samples : list[dict]
        Raw ``{"id","img","text","label"}`` samples.
    text_encoder : encoders.text_encoder.TextEncoder
        Loaded text encoder (reused across all samples).
    image_encoder : encoders.image_encoder.ImageEncoder
        Loaded image encoder (reused across all samples).

    Returns
    -------
    list
        ``torch_geometric.data.Data`` graphs carrying node features, edges and a
        ``y`` label.  Samples that fail to encode/build are skipped with a note.
    """
    from graph.graph_builder import build_graph
    from PIL import Image

    dataset: List[Any] = []
    total = len(samples)
    for i, sample in enumerate(samples):
        try:
            text = sample.get("text", "") or ""
            label = int(sample.get("label", 0))

            # --- Text elements. ---------------------------------------------
            text_out = text_encoder.encode(text)
            elements = list(text_out.get("elements", []))

            # --- Image elements (best-effort; skip image on failure). -------
            img_field = sample.get("img")
            if img_field is not None:
                try:
                    if not isinstance(img_field, str):
                        image = img_field
                    else:
                        # img paths in JSONL are relative (e.g. "img/42953.png");
                        # resolve against data_dir when provided.
                        img_path = img_field
                        if data_dir and not os.path.isabs(img_field):
                            img_path = os.path.join(data_dir, img_field)
                        image = Image.open(img_path).convert("RGB")
                    image_out = image_encoder.encode(image)
                    elements.extend(image_out.get("elements", []))
                except Exception as img_exc:  # pragma: no cover
                    print(
                        f"[CHEM][train] sample {sample.get('id', i)}: image "
                        f"encode failed ({img_exc}); using text only."
                    )

            if not elements:
                continue

            # --- Build the Content Composition Graph. -----------------------
            graph_dict = build_graph(elements, label)
            data = graph_dict["data"]
            dataset.append(data)
        except Exception as exc:  # pragma: no cover - never let one bad sample stop training
            print(f"[CHEM][train] skipping sample {sample.get('id', i)}: {exc}")

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"[CHEM][train] built graphs {i + 1}/{total}")

    return dataset


def _split_dataset(dataset: List[Any], seed: int = 42) -> Dict[str, List[Any]]:
    """Shuffle and split a graph list 70 / 15 / 15 into train / val / test.

    Parameters
    ----------
    dataset : list
        List of PyG ``Data`` graphs.
    seed : int, optional
        RNG seed for a reproducible shuffle (default 42).

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    import random

    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _compute_metrics(y_true: List[int], y_prob: List[float]) -> Dict[str, float]:
    """Compute accuracy / precision / recall / F1 / AUC for binary predictions.

    Uses ``evaluate.metrics.compute_metrics`` when available, otherwise falls
    back to scikit-learn directly.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth binary labels.
    y_prob : list[float]
        Predicted harmful-class probabilities.

    Returns
    -------
    dict
        ``{"accuracy","precision","recall","f1","auc"}`` (missing values 0.0).
    """
    try:
        from evaluate.metrics import compute_metrics

        return compute_metrics(y_true, y_prob)
    except Exception:
        pass  # fall through to sklearn

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0
    except Exception:  # pragma: no cover
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
    }


# --------------------------------------------------------------------------- #
# Train / evaluate loops
# --------------------------------------------------------------------------- #
def _compute_class_weights(dataset: List[Any], device) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle the harmful/safe imbalance.

    Returns a 2-element tensor [w_safe, w_harmful] where the minority class
    gets a higher weight so the loss treats both classes equally.
    """
    labels = [int(d.y.item()) for d in dataset]
    n = len(labels)
    n_harmful = sum(labels)
    n_safe = n - n_harmful
    # weight = total / (n_classes * count_per_class)
    w_safe    = n / (2.0 * max(n_safe, 1))
    w_harmful = n / (2.0 * max(n_harmful, 1))
    return torch.tensor([w_safe, w_harmful], dtype=torch.float, device=device)


def _evaluate(model, loader, device, criterion=None) -> Dict[str, Any]:
    """Run the model over a loader and return metrics + raw predictions.

    Parameters
    ----------
    model : CHEMGraphAttentionNetwork
        The model in any mode (set to eval internally).
    loader : torch_geometric.loader.DataLoader
        Batched graphs.
    device : torch.device
        Compute device.
    criterion : loss function, optional
        If None a plain NLLLoss is used (for backward compat).

    Returns
    -------
    dict
        ``{"metrics": {...}, "loss": float}``.
    """
    model.eval()
    if criterion is None:
        criterion = nn.NLLLoss()
    y_true: List[int] = []
    y_prob: List[float] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            log_probs = model(batch.x.float(), batch.edge_index, batch.batch)
            target = batch.y.view(-1).long()
            total_loss += float(criterion(log_probs, target).item())
            n_batches += 1

            probs = torch.exp(log_probs)[:, 1]  # harmful-class probability
            y_prob.extend(probs.cpu().tolist())
            y_true.extend(target.cpu().tolist())

    metrics = _compute_metrics(y_true, y_prob)
    return {"metrics": metrics, "loss": total_loss / max(n_batches, 1)}


def _train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """Train the model for a single epoch and return the mean loss.

    Parameters
    ----------
    model : CHEMGraphAttentionNetwork
        The model (set to train mode internally).
    loader : torch_geometric.loader.DataLoader
        Batched training graphs.
    optimizer : torch.optim.Optimizer
        The optimiser.
    criterion : torch.nn.Module
        Loss function (``NLLLoss``).
    device : torch.device
        Compute device.

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        log_probs = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = criterion(log_probs, batch.y.view(-1).long())
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train(
    epochs: int = 100,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    wandb_project: str = "chem",
) -> Optional[str]:
    """Run the full CHEM GAT training + evaluation pipeline.

    Parameters
    ----------
    epochs : int, optional
        Maximum number of epochs (early stopping may end sooner). Default 100.
    batch_size : int, optional
        Graphs per mini-batch (default 32).
    data_dir : str, optional
        Dataset root forwarded to the Hateful Memes loader.
    wandb_project : str, optional
        Weights & Biases project name (default ``"chem"``).

    Returns
    -------
    str or None
        Path to the best saved checkpoint, or ``None`` if no data was available
        to train on.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CHEM][train] Using device: {device}")

    # --- Weights & Biases (guarded so offline / missing wandb never crashes). -
    wandb = _init_wandb(wandb_project, epochs, batch_size)

    # --- Encoders loaded ONCE and reused for every sample. ------------------
    print("[CHEM][train] Loading text + image encoders (once) ...")
    from encoders.text_encoder import TextEncoder
    from encoders.image_encoder import ImageEncoder

    text_encoder = TextEncoder(device=str(device))
    image_encoder = ImageEncoder(device=str(device))

    # --- Build the dataset. -------------------------------------------------
    print("[CHEM][train] Loading Hateful Memes samples ...")
    samples = _load_samples("train", data_dir)
    # Some distributions ship a separate dev split; fold it in when present.
    samples += _load_samples("dev", data_dir)
    print(f"[CHEM][train] Loaded {len(samples)} raw samples.")

    if not samples:
        print("[CHEM][train] ERROR: no samples available; aborting training.")
        return None

    dataset = _build_dataset(samples, text_encoder, image_encoder)
    print(f"[CHEM][train] Built {len(dataset)} content-composition graphs.")
    if not dataset:
        print("[CHEM][train] ERROR: no graphs built; aborting training.")
        return None

    splits = _split_dataset(dataset)
    print(
        f"[CHEM][train] Split -> train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )

    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(splits["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(splits["test"], batch_size=batch_size, shuffle=False)

    # --- Class weights: upweight the minority (harmful) class. -------------
    class_weights = _compute_class_weights(splits["train"], device)
    n_harmful = int(class_weights.argmin().item() == 0)  # for logging only
    print(
        f"[CHEM][train] Class weights -> safe={class_weights[0]:.3f} "
        f"harmful={class_weights[1]:.3f}"
    )

    # --- Model / optimiser / loss. -----------------------------------------
    model = CHEMGraphAttentionNetwork().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY
    )
    # Halve LR when val F1 stops improving — helps escape plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=_LR_SCHEDULE_PATIENCE
    )
    criterion = nn.NLLLoss(weight=class_weights)

    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

    # --- Training loop with early stopping on validation F1. ---------------
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    print("[CHEM][train] Starting training ...")
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_eval = _evaluate(model, val_loader, device, criterion)
        val_metrics = val_eval["metrics"]
        val_f1 = val_metrics.get("f1", 0.0)

        print(
            f"[CHEM][train] epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_eval['loss']:.4f} | "
            f"val_f1={val_f1:.4f} | val_acc={val_metrics.get('accuracy', 0.0):.4f}"
        )

        _wandb_log(
            wandb,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_eval["loss"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            },
        )

        # LR scheduler step (reduce on plateau).
        scheduler.step(val_f1)

        # Checkpoint + early-stopping bookkeeping.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {"state_dict": model.state_dict(), "val_f1": best_val_f1, "epoch": epoch},
                _BEST_CHECKPOINT_PATH,
            )
            print(
                f"[CHEM][train] new best val_f1={best_val_f1:.4f}; "
                f"saved -> {_BEST_CHECKPOINT_PATH}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= _EARLY_STOP_PATIENCE:
                print(
                    f"[CHEM][train] early stopping at epoch {epoch} "
                    f"(no val F1 improvement in {_EARLY_STOP_PATIENCE} epochs)."
                )
                break

    # --- Final test-set evaluation using the best checkpoint. --------------
    print("[CHEM][train] Loading best checkpoint for test evaluation ...")
    if os.path.isfile(_BEST_CHECKPOINT_PATH):
        checkpoint = torch.load(_BEST_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    test_eval = _evaluate(model, test_loader, device)
    test_metrics = test_eval["metrics"]
    print("[CHEM][train] ===== TEST METRICS =====")
    print(f"[CHEM][train] accuracy : {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"[CHEM][train] precision: {test_metrics.get('precision', 0.0):.4f}")
    print(f"[CHEM][train] recall   : {test_metrics.get('recall', 0.0):.4f}")
    print(f"[CHEM][train] f1       : {test_metrics.get('f1', 0.0):.4f}")
    print(f"[CHEM][train] auc      : {test_metrics.get('auc', 0.0):.4f}")

    _wandb_log(wandb, {f"test_{k}": v for k, v in test_metrics.items()})
    _wandb_finish(wandb)

    return _BEST_CHECKPOINT_PATH


# --------------------------------------------------------------------------- #
# Weights & Biases helpers (graceful degradation)
# --------------------------------------------------------------------------- #
def _init_wandb(project: str, epochs: int, batch_size: int):
    """Initialise a W&B run, honouring ``WANDB_MODE`` and failing soft.

    Parameters
    ----------
    project : str
        W&B project name.
    epochs, batch_size : int
        Hyper-parameters recorded in the run config.

    Returns
    -------
    module or None
        The ``wandb`` module if a run started, else ``None`` (training then runs
        without logging).
    """
    try:
        import wandb

        mode = os.environ.get("WANDB_MODE", "online")
        wandb.init(
            project=project,
            mode=mode,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": _LEARNING_RATE,
                "weight_decay": _WEIGHT_DECAY,
                "early_stop_patience": _EARLY_STOP_PATIENCE,
                "architecture": "GAT(768->256x8->2)",
            },
        )
        print(f"[CHEM][train] Weights & Biases initialised (mode={mode}).")
        return wandb
    except Exception as exc:  # pragma: no cover - wandb optional
        print(f"[CHEM][train] W&B unavailable ({exc}); continuing without logging.")
        return None


def _wandb_log(wandb, payload: Dict[str, Any]) -> None:
    """Log a metric dict to W&B if a run is active (no-op otherwise).

    Parameters
    ----------
    wandb : module or None
        The active wandb module, or None.
    payload : dict
        Metrics to log.

    Returns
    -------
    None
    """
    if wandb is None:
        return
    try:
        wandb.log(payload)
    except Exception:  # pragma: no cover
        pass


def _wandb_finish(wandb) -> None:
    """Finish the active W&B run if any (no-op otherwise).

    Parameters
    ----------
    wandb : module or None
        The active wandb module, or None.

    Returns
    -------
    None
    """
    if wandb is None:
        return
    try:
        wandb.finish()
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """Parse CLI arguments and launch training.

    Inputs: command-line arguments (``--epochs``, ``--batch_size``,
    ``--data_dir``, ``--wandb_project``).
    Outputs: none (side effect: trains + saves a checkpoint, prints metrics).
    """
    parser = argparse.ArgumentParser(description="Train the CHEM GAT harm classifier.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Graphs per batch.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Hateful Memes dataset root (forwarded to the loader).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="chem",
        help="Weights & Biases project name.",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
