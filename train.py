"""
Training script for ISICSkinModel on ISIC 2019.

Optimized for NVIDIA A40 (48 GB) on RunPod:
  - FP16 AMP + GradScaler for 2x throughput
  - batch_size=32, no accumulation needed
  - torch.compile for extra speed
  - 8 DataLoader workers
  - Cosine LR with linear warm-up
  - Class-weighted CE + label smoothing (0.1)
  - Heavy augmentation (RandAugment, erasing, flips, jitter)
  - Weighted random sampling for class imbalance
  - Stratified 85/15 train/val split

Also works on MPS (float32) and CPU.

Usage (RunPod A40):
    python train.py --device cuda:0 --batch_size 32 --accum_steps 1 --workers 8 --epochs 20 --compile --mixup

Usage (local Mac):
    python train.py --device mps --batch_size 4 --accum_steps 8 --workers 4 --epochs 5 --freeze_backbone
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from dataset import CLASSES, ISICDataset, compute_class_weights
from model import build_isic_model

# Module-level logger — configured in main() via setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Paths                                                              #
# ------------------------------------------------------------------ #
DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGE_DIR = DATA_DIR / "ISIC_2019_Training_Input"
GT_CSV = DATA_DIR / "ISIC_2019_Training_GroundTruth.csv"
META_CSV = DATA_DIR / "ISIC_2019_Training_Metadata.csv"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

# ------------------------------------------------------------------ #
#  Defaults (tuned for A40 48 GB)                                     #
# ------------------------------------------------------------------ #
IMG_SIZE = 384
NUM_CLASSES = 8
METADATA_DIM = 13
EPOCHS = 25
BATCH_SIZE = 48          # A40: ~27 GB VRAM at 384x384 FP16
ACCUM_STEPS = 1
LR = 4e-4
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 2
LABEL_SMOOTH = 0.1
NUM_WORKERS = 12
SEED = 42
VAL_FRAC = 0.15
MIXUP_ALPHA = 0.2
PREFETCH_FACTOR = 4      # pipeline data loading
LOG_DIR = Path(__file__).resolve().parent / "logs"


# ------------------------------------------------------------------ #
#  Logging setup                                                      #
# ------------------------------------------------------------------ #
def setup_logging(run_name: str = "train") -> None:
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

    logger.info("Run log saved to: %s", log_file)


# ------------------------------------------------------------------ #
#  Reproducibility                                                    #
# ------------------------------------------------------------------ #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------ #
#  Transforms                                                         #
# ------------------------------------------------------------------ #
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


# ------------------------------------------------------------------ #
#  Mixup                                                              #
# ------------------------------------------------------------------ #
def mixup_data(
    images: torch.Tensor,
    meta: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_meta = lam * meta + (1 - lam) * meta[index]
    return mixed_images, mixed_meta, labels, labels[index], lam


def mixup_criterion(
    criterion: nn.Module, logits: torch.Tensor,
    y_a: torch.Tensor, y_b: torch.Tensor, lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ------------------------------------------------------------------ #
#  LR schedule                                                        #
# ------------------------------------------------------------------ #
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
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------ #
#  Weighted sampler                                                   #
# ------------------------------------------------------------------ #
def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


# ------------------------------------------------------------------ #
#  Validation                                                         #
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, float, float, dict]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            logits = model(images, meta)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(all_labels)
    top1_acc = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    per_class_acc = {}
    for c in range(NUM_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[CLASSES[c]] = float((all_preds[mask] == all_labels[mask]).mean())
        else:
            per_class_acc[CLASSES[c]] = 0.0

    return avg_loss, top1_acc, bal_acc, per_class_acc


# ------------------------------------------------------------------ #
#  Train one epoch                                                    #
# ------------------------------------------------------------------ #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    amp_dtype: torch.dtype,
    accum_steps: int,
    epoch: int,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_scaler = scaler is not None and scaler.is_enabled()

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False, dynamic_ncols=True)
    for step, (images, meta, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup (skip first 2 epochs to let heads warm up)
        if use_mixup and epoch >= 2:
            mixed_img, mixed_meta, y_a, y_b, lam = mixup_data(images, meta, labels, mixup_alpha)
        else:
            mixed_img, mixed_meta, y_a, y_b, lam = images, meta, labels, labels, 1.0

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            logits = model(mixed_img, mixed_meta)
            if use_mixup and epoch >= 2:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam) / accum_steps
            else:
                loss = criterion(logits, labels) / accum_steps

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * accum_steps * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step % 50 == 0:
            pbar.set_postfix(
                loss=f"{total_loss/total:.4f}",
                acc=f"{correct/total:.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return total_loss / total, correct / total


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train ISICSkinModel")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum_steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                        help="Freeze backbone, train only MLP heads")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile (CUDA, PyTorch 2+)")
    parser.add_argument("--mixup", action="store_true", default=False,
                        help="Enable Mixup augmentation")
    parser.add_argument("--no_weighted_sampling", action="store_true", default=False,
                        help="Disable weighted random sampler")
    args = parser.parse_args()

    seed_everything(SEED)
    setup_logging("train")
    logger.info("Args: %s", json.dumps(vars(args), indent=2))

    # ---- Device -------------------------------------------------- #
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        amp_dtype = torch.float16
    elif device.type == "mps":
        amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    sep = "=" * 60
    logger.info(sep)
    logger.info("  ISICSkinModel Training")
    logger.info(sep)
    logger.info("  Device       : %s", device)
    if device.type == "cuda":
        logger.info("  GPU          : %s", torch.cuda.get_device_name(device))
        logger.info("  VRAM         : %.1f GB", torch.cuda.get_device_properties(device).total_memory / 1e9)
    logger.info("  AMP dtype    : %s", amp_dtype)
    logger.info("  Batch size   : %d x %d = %d", args.batch_size, args.accum_steps, args.batch_size * args.accum_steps)
    logger.info("  LR           : %g", args.lr)
    logger.info("  Epochs       : %d", args.epochs)
    logger.info("  Freeze bb    : %s", args.freeze_backbone)
    logger.info("  Compile      : %s", args.compile)
    logger.info("  Mixup        : %s", args.mixup)
    logger.info(sep)

    # ---- Data ---------------------------------------------------- #
    gt = pd.read_csv(GT_CSV)
    labels_all = gt[CLASSES].values.argmax(axis=1)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels_all)), labels_all))

    train_ds = ISICDataset(
        str(IMAGE_DIR), str(GT_CSV), str(META_CSV),
        transform=get_train_transform(), indices=train_idx,
    )
    val_ds = ISICDataset(
        str(IMAGE_DIR), str(GT_CSV), str(META_CSV),
        transform=get_val_transform(), indices=val_idx,
    )

    logger.info("  Train samples: %d", len(train_ds))
    logger.info("  Val samples  : %d", len(val_ds))

    # Weighted sampler
    use_weighted = not args.no_weighted_sampling
    if use_weighted:
        train_sampler = make_weighted_sampler(train_ds.labels)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    pin = device.type == "cuda"
    prefetch = PREFETCH_FACTOR if args.workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=pin,
        drop_last=True, persistent_workers=(args.workers > 0),
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=(args.workers > 0),
        prefetch_factor=prefetch,
    )

    # ---- Model --------------------------------------------------- #
    model = build_isic_model(
        num_classes=NUM_CLASSES,
        metadata_dim=METADATA_DIM,
        in_chans=3,
        pretrained=args.pretrained,
    ).to(device)

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("  Backbone FROZEN")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("  Params       : %.1fM total, %.1fM trainable", total_params, train_params)

    if args.compile and device.type == "cuda":
        logger.info("  Compiling model ...")
        model = torch.compile(model)

    # ---- Loss ---------------------------------------------------- #
    class_weights = compute_class_weights(str(GT_CSV)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    # ---- Optimizer & Scheduler ----------------------------------- #
    if args.freeze_backbone:
        opt_params = [
            {"params": model.metadata_mlp.parameters(), "lr": args.lr},
            {"params": model.fusion_head.parameters(), "lr": args.lr},
        ]
    else:
        backbone_params = list(model.backbone.parameters())
        head_params = (
            list(model.metadata_mlp.parameters()) +
            list(model.fusion_head.parameters())
        )
        opt_params = [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ]

    optimizer = torch.optim.AdamW(opt_params, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    # ---- Resume -------------------------------------------------- #
    start_epoch = 0
    best_val_acc = 0.0
    best_bal_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_bal_acc = ckpt.get("best_bal_acc", 0.0)
        logger.info("Resumed from epoch %d, best_val_acc=%.4f", start_epoch, best_val_acc)

    # ---- Training Loop ------------------------------------------- #
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting training — %d epochs", args.epochs)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, amp_dtype, args.accum_steps, epoch,
            use_mixup=args.mixup, mixup_alpha=MIXUP_ALPHA,
        )

        val_loss, val_acc, bal_acc, per_class = validate(
            model, val_loader, criterion, device, amp_dtype,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - t0

        logger.info(
            "Epoch %2d/%d | Train Loss=%.4f Acc=%.4f | Val Loss=%.4f Acc=%.4f BalAcc=%.4f | %.0fs",
            epoch + 1, args.epochs, train_loss, train_acc, val_loss, val_acc, bal_acc, elapsed,
        )
        pc_str = "  ".join(f"{cls}={v:.3f}" for cls, v in per_class.items())
        logger.info("  [Per-class] %s", pc_str)

        is_best = val_acc > best_val_acc
        is_best_bal = bal_acc > best_bal_acc
        if is_best:
            best_val_acc = val_acc
        if is_best_bal:
            best_bal_acc = bal_acc

        state = {
            "epoch": epoch,
            "model_state_dict": (model._orig_mod.state_dict()
                                 if hasattr(model, "_orig_mod") else model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "best_bal_acc": best_bal_acc,
            "val_acc": val_acc,
            "bal_acc": bal_acc,
            "val_loss": val_loss,
        }
        torch.save(state, CKPT_DIR / "last.pt")
        if is_best:
            torch.save(state, CKPT_DIR / "best.pt")
            logger.info("** New best val acc: %.4f **", best_val_acc)
        if is_best_bal:
            torch.save(state, CKPT_DIR / "best_balanced.pt")
            logger.info("** New best balanced acc: %.4f **", best_bal_acc)

    sep = "=" * 60
    logger.info(sep)
    logger.info("  Training complete!")
    logger.info("  Best val accuracy     : %.4f", best_val_acc)
    logger.info("  Best balanced accuracy: %.4f", best_bal_acc)
    logger.info("  Checkpoints: %s", CKPT_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
