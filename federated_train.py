"""
Federated training script for ISICSkinModel with structure-aware
personalized aggregation.

Aggregation strategy (per round):
  Group A  (patch_embed + stages 0–1)  → plain FedAvg
  Group B  (stages 2–3 + norm)         → drift-aware weighted avg  (or FedAvg)
  Group C  (metadata_mlp + fusion)     → local only (never aggregated)

Supports:
  - Dirichlet-based non-IID partition (--dirichlet_alpha)
  - Fractional client participation   (--client_frac)
  - AMP / GradScaler on CUDA
  - Resume from federated checkpoint
  - Per-round logging & checkpointing

Usage (RunPod A40):
    python federated_train.py \\
        --device cuda:0 --num_clients 5 --rounds 50 \\
        --local_epochs 3 --batch_size 32 --dirichlet_alpha 0.5

Usage (local Mac):
    python federated_train.py \\
        --device mps --num_clients 3 --rounds 10 \\
        --local_epochs 2 --batch_size 4 --dirichlet_alpha 1.0
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from dataset import CLASSES, ISICDataset, compute_class_weights
from federated_utils import (
    apply_params,
    dirichlet_partition,
    drift_aware_aggregate,
    fedavg,
    split_model_parameters,
    validate_partition,
)
from model import build_isic_model

# Module-level logger — configured in main() via setup_logging()
logger = logging.getLogger(__name__)

# ====================================================================== #
#  Paths                                                                  #
# ====================================================================== #
DATA_DIR  = Path(__file__).resolve().parent / "data"
IMAGE_DIR = DATA_DIR / "ISIC_2019_Training_Input"
GT_CSV    = DATA_DIR / "ISIC_2019_Training_GroundTruth.csv"
META_CSV  = DATA_DIR / "ISIC_2019_Training_Metadata.csv"
CKPT_DIR  = Path(__file__).resolve().parent / "checkpoints" / "federated"

# ====================================================================== #
#  Defaults (same hypers as centralized train.py unless overridden)       #
# ====================================================================== #
IMG_SIZE      = 384
NUM_CLASSES   = 8
METADATA_DIM  = 13
LR            = 4e-4
WEIGHT_DECAY  = 1e-2
WARMUP_EPOCHS = 2
LABEL_SMOOTH  = 0.1
SEED          = 42
VAL_FRAC      = 0.15
NUM_WORKERS   = 8
PREFETCH      = 4
BATCH_SIZE    = 32
EPSILON       = 1e-6
LOG_DIR       = Path(__file__).resolve().parent / "logs"


# ====================================================================== #
#  Logging setup                                                          #
# ====================================================================== #
def setup_logging(run_name: str = "federated") -> None:
    """
    Configure the module logger to write to both the console and a
    timestamped log file under LOG_DIR.

    Format:  [2026-02-23 14:05:00] [INFO    ] message
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOG_DIR / f"{run_name}_{timestamp}.log"

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()              # console
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")  # file
    fh.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()                   # avoid duplicate handlers on re-run
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    logger.info(f"Run log saved to: {log_file}")


# ====================================================================== #
#  Reproducibility                                                        #
# ====================================================================== #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ====================================================================== #
#  Transforms (identical to train.py)                                     #
# ====================================================================== #
def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(45),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


def get_val_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ====================================================================== #
#  Weighted sampler for a subset of labels                                #
# ====================================================================== #
def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


# ====================================================================== #
#  Cosine schedule with linear warm-up (per-client)                       #
# ====================================================================== #
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1 + math.cos(math.pi * progress)
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ====================================================================== #
#  Local training for a single client                                     #
# ====================================================================== #
def train_client(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    local_epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[OrderedDict, Dict]:
    """
    Train *model* locally for *local_epochs* on *loader*.

    Returns the updated ``state_dict`` (on CPU to save GPU memory).
    """
    model.train()

    # Build optimizer fresh each round (stateless across rounds — standard FL)
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.metadata_mlp.parameters())
        + list(model.fusion_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * local_epochs
    warmup_steps = steps_per_epoch * min(WARMUP_EPOCHS, local_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler() if use_scaler else None

    # Metric accumulators
    total_loss = 0.0
    all_probs: list = []
    all_preds: list = []
    all_labels: list = []

    for epoch in range(local_epochs):
        for images, meta, labels in loader:
            images = images.to(device, non_blocking=True)
            meta   = meta.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device.type, dtype=amp_dtype,
                enabled=(amp_dtype != torch.float32),
            ):
                logits = model(images, meta)
                loss = criterion(logits, labels)

            if use_scaler:
                scaler.scale(loss).backward()           # type: ignore[union-attr]
                scaler.unscale_(optimizer)               # type: ignore[union-attr]
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)                   # type: ignore[union-attr]
                scaler.update()                          # type: ignore[union-attr]
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # --- accumulate metrics for this batch -----------------
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            with torch.no_grad():
                probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
                preds_batch = probs_batch.argmax(axis=1)
            all_probs.append(probs_batch)
            all_preds.append(preds_batch)
            all_labels.append(labels.cpu().numpy())

            scheduler.step()

    # Return state-dict on CPU to avoid holding many copies on GPU
    preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([], dtype=int)
    labs = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([], dtype=int)
    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.zeros((0, NUM_CLASSES))

    train_loss = total_loss / max(1, len(labs)) if labs.size > 0 else 0.0
    acc = float((preds == labs).mean()) if labs.size > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(labs, preds)) if labs.size > 0 else 0.0

    p, r, f1, _ = precision_recall_fscore_support(labs, preds, average="macro", zero_division=0) if labs.size > 0 else (0.0, 0.0, 0.0, None)

    # AUC / AP (multiclass) — may fail if a class has no positives
    try:
        y_true_oh = np.eye(NUM_CLASSES, dtype=np.int64)[labs]
        roc_auc = float(roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr")) if probs.size > 0 else float("nan")
    except Exception:
        roc_auc = float("nan")

    try:
        ap = float(average_precision_score(y_true_oh, probs, average="macro")) if probs.size > 0 else float("nan")
    except Exception:
        ap = float("nan")

    metrics = {
        "train_loss": train_loss,
        "acc": acc,
        "bal_acc": bal_acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "avg_precision": ap,
        "num_samples": int(labs.size),
    }

    state = OrderedDict((k, v.cpu()) for k, v in model.state_dict().items())
    return state, metrics


# ====================================================================== #
#  Validation helper (mirrors train.py)                                   #
# ====================================================================== #
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict:
    """
    Evaluate *model* on *loader*.

    Returns dict with keys: loss, acc, bal_acc, per_class.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta   = meta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)
        ):
            logits = model(images, meta)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_labels.append(labels.cpu().numpy())

    preds  = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([], dtype=int)
    labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([], dtype=int)
    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.zeros((0, NUM_CLASSES))
    n = len(labels)

    avg_loss = total_loss / max(1, n)
    acc = float((preds == labels).mean()) if n > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(labels, preds)) if n > 0 else 0.0

    per_class = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        per_class[CLASSES[c]] = float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else 0.0

    # Precision / Recall / F1 (macro)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0) if n > 0 else (0.0, 0.0, 0.0, None)

    # AUC / AP (multiclass) — robust handling
    try:
        y_true_oh = np.eye(NUM_CLASSES, dtype=np.int64)[labels]
        roc_auc = float(roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr")) if probs.size > 0 else float("nan")
    except Exception:
        roc_auc = float("nan")

    try:
        ap = float(average_precision_score(y_true_oh, probs, average="macro")) if probs.size > 0 else float("nan")
    except Exception:
        ap = float("nan")

    return {
        "loss": avg_loss,
        "acc": acc,
        "bal_acc": bal_acc,
        "per_class": per_class,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "avg_precision": ap,
    }


# ====================================================================== #
#  Main federated loop                                                    #
# ====================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated training: structure-aware personalized aggregation"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--client_frac", type=float, default=1.0,
                        help="Fraction of clients participating each round")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5,
                        help="Dirichlet α for non-IID partition (smaller → more heterogeneous)")
    parser.add_argument("--drift_weighting", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=True,
                        help="Use drift-aware aggregation for Group B (default: True)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to federated checkpoint to resume from")
    parser.add_argument("--pretrained", action="store_true", default=True)
    args = parser.parse_args()

    seed_everything(SEED)
    setup_logging("federated")
    logger.info("Args: %s", json.dumps(vars(args), indent=2))

    # ------------------------------------------------------------------ #
    #  Device & AMP                                                       #
    # ------------------------------------------------------------------ #
    if args.device:
        # Honour user request but validate availability for CUDA/MPS
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested device %s not available; falling back to CPU", args.device)
            device = torch.device("cpu")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("Requested device %s not available; falling back to CPU", args.device)
            device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # A40 uses float16 (Ampere supports it well; bfloat16 is for A100/H100/L40S)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # ------------------------------------------------------------------ #
    #  Banner                                                             #
    # ------------------------------------------------------------------ #
    sep = "=" * 66
    logger.info(sep)
    logger.info("  Structure-Aware Personalized Federated Learning — SwinV2")
    logger.info(sep)
    logger.info("  Device           : %s", device)
    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(device)
            logger.info("  GPU              : %s", props.name)
            logger.info("  VRAM             : %.1f GB", props.total_memory / 1e9)
            logger.info("  SM count         : %d", props.multi_processor_count)
            logger.info("  BF16 native      : %s", torch.cuda.is_bf16_supported())
        except Exception as e:
            logger.warning("  GPU info unavailable: %s", e)
    logger.info("  AMP dtype        : %s", amp_dtype)
    logger.info("  Clients          : %d", args.num_clients)
    logger.info("  Rounds           : %d", args.rounds)
    logger.info("  Local epochs     : %d", args.local_epochs)
    logger.info("  Client fraction  : %.2f", args.client_frac)
    logger.info("  Dirichlet α      : %.3f", args.dirichlet_alpha)
    logger.info("  Drift weighting  : %s", args.drift_weighting)
    logger.info("  Batch size       : %d", args.batch_size)
    logger.info("  LR               : %g", args.lr)
    logger.info(sep)

    # ------------------------------------------------------------------ #
    #  Data — stratified train/val split, then Dirichlet partition train  #
    # ------------------------------------------------------------------ #
    gt = pd.read_csv(GT_CSV)
    labels_all = gt[CLASSES].values.argmax(axis=1)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels_all)), labels_all))

    # Full datasets (transforms applied per-client)
    train_ds = ISICDataset(
        str(IMAGE_DIR), str(GT_CSV), str(META_CSV),
        transform=get_train_transform(), indices=train_idx,
    )
    val_ds = ISICDataset(
        str(IMAGE_DIR), str(GT_CSV), str(META_CSV),
        transform=get_val_transform(), indices=val_idx,
    )

    # --- Dirichlet non-IID partition of training set ------------------ #
    for attempt in range(50):
        client_partitions = dirichlet_partition(
            train_ds.labels, args.num_clients, args.dirichlet_alpha,
            seed=SEED + attempt,
        )
        if validate_partition(train_ds.labels, client_partitions):
            if attempt > 0:
                logger.info("  Valid partition found on attempt %d (seed=%d)", attempt + 1, SEED + attempt)
            break
    else:
        raise RuntimeError(
            f"Failed to generate valid Dirichlet partition after 50 attempts "
            f"(α={args.dirichlet_alpha}, clients={args.num_clients}). "
            f"Try increasing --dirichlet_alpha or decreasing --num_clients."
        )

    for k, part in enumerate(client_partitions):
        dist = np.bincount(train_ds.labels[part], minlength=NUM_CLASSES)
        logger.info("  Client %d: %5d samples  class-dist=%s", k, len(part), dist.tolist())

    # --- Validation loader (global, used every round) ----------------- #
    pin = device.type == "cuda"
    prefetch = PREFETCH if args.workers > 0 else None
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=(args.workers > 0),
        prefetch_factor=prefetch,
    )

    # ------------------------------------------------------------------ #
    #  Model & parameter groups                                           #
    # ------------------------------------------------------------------ #
    global_model = build_isic_model(
        num_classes=NUM_CLASSES,
        metadata_dim=METADATA_DIM,
        in_chans=3,
        pretrained=args.pretrained,
    ).to(device)

    param_groups = split_model_parameters(global_model)
    logger.info("  Group A (global):  %3d param tensors", len(param_groups['group_A']))
    logger.info("  Group B (drift):   %3d param tensors", len(param_groups['group_B']))
    logger.info("  Group C (local):   %3d param tensors", len(param_groups['group_C']))
    total_params = sum(p.numel() for p in global_model.parameters()) / 1e6
    logger.info("  Total params:      %.1fM", total_params)

    # --- Loss (same as centralized) ----------------------------------- #
    class_weights = compute_class_weights(str(GT_CSV)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    # ------------------------------------------------------------------ #
    #  Per-client personalized state (Group C kept independently)         #
    # ------------------------------------------------------------------ #
    # Initialise every client with the same starting state
    client_states: List[OrderedDict] = [
        OrderedDict((k, v.cpu().clone()) for k, v in global_model.state_dict().items())
        for _ in range(args.num_clients)
    ]

    # ------------------------------------------------------------------ #
    #  Resume                                                             #
    # ------------------------------------------------------------------ #
    start_round = 0
    best_global_bal_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        global_model.load_state_dict(ckpt["global_model_state"])
        client_states = ckpt["client_states"]
        start_round = ckpt["round"] + 1
        best_global_bal_acc = ckpt.get("best_global_bal_acc", 0.0)
        logger.info("Resumed from round %d, best_bal_acc=%.4f", start_round, best_global_bal_acc)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Federated training loop                                            #
    # ------------------------------------------------------------------ #
    logger.info("Starting federated training — %d rounds, %d clients", args.rounds, args.num_clients)

    for rnd in range(start_round, args.rounds):
        t0 = time.time()

        # --- 1. Select participating clients -------------------------- #
        num_participate = max(1, int(args.num_clients * args.client_frac))
        participating = sorted(
            random.sample(range(args.num_clients), num_participate)
        )

        # --- 2. Broadcast global params (A+B) to selected clients ----- #
        global_sd = OrderedDict(
            (k, v.cpu().clone()) for k, v in global_model.state_dict().items()
        )

        updated_client_sds: List[OrderedDict] = []
        client_dataset_sizes: List[int] = []

        for k in participating:
            # Build client-specific state:
            #   Groups A & B ← global model (broadcast)
            #   Group C      ← client's own local state  (personalised)
            client_sd = OrderedDict()
            for name in global_sd:
                if name in param_groups["group_C"]:
                    client_sd[name] = client_states[k][name].clone()
                else:
                    client_sd[name] = global_sd[name].clone()

            # Instantiate a fresh model on device for local training
            client_model = build_isic_model(
                num_classes=NUM_CLASSES,
                metadata_dim=METADATA_DIM,
                in_chans=3,
                pretrained=False,
            )
            client_model.load_state_dict(client_sd)
            client_model.to(device)

            # Client data loader (with weighted sampling for class balance)
            part_indices = client_partitions[k]
            client_subset = Subset(train_ds, part_indices)
            client_labels = train_ds.labels[part_indices]
            sampler = make_weighted_sampler(client_labels)

            client_loader = DataLoader(
                client_subset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=min(args.workers, 4),
                pin_memory=pin,
                drop_last=True,
                prefetch_factor=prefetch if min(args.workers, 4) > 0 else None,
            )

            # --- 3. Local training ------------------------------------ #
            updated_sd, train_metrics = train_client(
                model=client_model,
                loader=client_loader,
                criterion=criterion,
                device=device,
                amp_dtype=amp_dtype,
                local_epochs=args.local_epochs,
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
            )
            updated_client_sds.append(updated_sd)
            client_dataset_sizes.append(len(part_indices))

            # Persist full client state (including updated Group C)
            client_states[k] = updated_sd

            # Log per-client training summary
            logger.info(
                "  [Train] Client %d | samples=%d loss=%.4f acc=%.4f "
                "bal_acc=%.4f f1=%.4f roc_auc=%.4f",
                k, train_metrics['num_samples'], train_metrics['train_loss'],
                train_metrics['acc'], train_metrics['bal_acc'],
                train_metrics['f1'], train_metrics['roc_auc'],
            )

            # Free GPU memory from this client's model
            del client_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # --- 4. Server aggregation ------------------------------------ #
        # Group A: plain FedAvg (weighted by dataset size)
        agg_A = fedavg(
            updated_client_sds,
            param_groups["group_A"],
            weights=client_dataset_sizes,
        )

        # Group B: drift-aware or plain FedAvg
        if args.drift_weighting:
            agg_B = drift_aware_aggregate(
                updated_client_sds,
                param_groups["group_B"],
                epsilon=EPSILON,
            )
        else:
            agg_B = fedavg(
                updated_client_sds,
                param_groups["group_B"],
                weights=client_dataset_sizes,
            )

        # Apply aggregated Groups A+B to global model
        new_global_sd = global_model.state_dict()
        for name, param in agg_A.items():
            new_global_sd[name] = param.to(new_global_sd[name].device)
        for name, param in agg_B.items():
            new_global_sd[name] = param.to(new_global_sd[name].device)
        global_model.load_state_dict(new_global_sd)

        # --- 5. Global evaluation ------------------------------------- #
        global_metrics = validate(global_model, val_loader, criterion, device, amp_dtype)

        # --- 6. Per-client evaluation (for worst-client / std) -------- #
        client_accs: List[float] = []
        client_bal_accs: List[float] = []
        for k in participating:
            # Evaluate client k with its personalized head (Group C)
            eval_model = build_isic_model(
                num_classes=NUM_CLASSES,
                metadata_dim=METADATA_DIM,
                in_chans=3,
                pretrained=False,
            )
            # Client state: Groups A+B from new global, Group C from client
            eval_sd = OrderedDict()
            new_global_cpu = OrderedDict(
                (n, v.cpu()) for n, v in global_model.state_dict().items()
            )
            for name in new_global_cpu:
                if name in param_groups["group_C"]:
                    eval_sd[name] = client_states[k][name]
                else:
                    eval_sd[name] = new_global_cpu[name]
            eval_model.load_state_dict(eval_sd)
            eval_model.to(device)

            cm = validate(eval_model, val_loader, criterion, device, amp_dtype)
            client_accs.append(cm["acc"])
            client_bal_accs.append(cm["bal_acc"])

            del eval_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        worst_acc     = min(client_accs)
        std_acc       = float(np.std(client_accs))
        worst_bal_acc = min(client_bal_accs)
        std_bal_acc   = float(np.std(client_bal_accs))

        elapsed = time.time() - t0

        # --- 7. Logging ----------------------------------------------- #
        logger.info(
            "Round %3d/%d | Global Acc=%.4f BalAcc=%.4f F1=%.4f AUC=%.4f | "
            "Worst Acc=%.4f BalAcc=%.4f | Std Acc=%.4f BalAcc=%.4f | %.0fs",
            rnd + 1, args.rounds,
            global_metrics['acc'], global_metrics['bal_acc'],
            global_metrics.get('f1', 0.0), global_metrics.get('roc_auc', float('nan')),
            worst_acc, worst_bal_acc, std_acc, std_bal_acc, elapsed,
        )
        pc = global_metrics["per_class"]
        pc_str = "  ".join(f"{cls}={v:.3f}" for cls, v in pc.items())
        logger.info("  [Per-class] %s", pc_str)

        # --- 8. Checkpointing ---------------------------------------- #
        is_best = global_metrics["bal_acc"] > best_global_bal_acc
        if is_best:
            best_global_bal_acc = global_metrics["bal_acc"]

        ckpt_state = {
            "round": rnd,
            "global_model_state": OrderedDict(
                (n, v.cpu()) for n, v in global_model.state_dict().items()
            ),
            "client_states": client_states,
            "param_groups": {g: list(s) for g, s in param_groups.items()},
            "best_global_bal_acc": best_global_bal_acc,
            "args": vars(args),
            "global_metrics": global_metrics,
            "client_accs": client_accs,
            "client_bal_accs": client_bal_accs,
        }

        torch.save(ckpt_state, CKPT_DIR / f"global_round_{rnd+1}.pt")
        torch.save(ckpt_state, CKPT_DIR / "last_federated.pt")

        if is_best:
            torch.save(ckpt_state, CKPT_DIR / "best_federated.pt")
            logger.info("** New best global balanced accuracy: %.4f **", best_global_bal_acc)

        # Save final personalized client models at last round
        if rnd == args.rounds - 1:
            for k in range(args.num_clients):
                client_ckpt = {
                    "round": rnd,
                    "client_id": k,
                    "client_state": client_states[k],
                }
                torch.save(client_ckpt, CKPT_DIR / f"client_{k}_final.pt")
            logger.info("Saved %d personalized client models to %s", args.num_clients, CKPT_DIR)

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #
    sep = "=" * 66
    logger.info(sep)
    logger.info("  Federated training complete!")
    logger.info("  Best global balanced accuracy: %.4f", best_global_bal_acc)
    logger.info("  Checkpoints: %s", CKPT_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
