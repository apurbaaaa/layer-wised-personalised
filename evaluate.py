"""
Evaluate the best checkpoint on the ISIC 2019 test set.

Usage:
    python evaluate.py                                     # uses best.pt
    python evaluate.py --checkpoint checkpoints/last.pt    # specify checkpoint
    python evaluate.py --split val                         # evaluate on val split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CLASSES, ISICDataset
from model import build_isic_model

DATA_DIR = Path(__file__).resolve().parent / "data"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

IMG_SIZE = 384
NUM_CLASSES = 8
METADATA_DIM = 13


def get_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms() -> list[transforms.Compose]:
    """Test-time augmentation: original + HFlip + VFlip + HFlip+VFlip."""
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm]

    return [
        transforms.Compose(base),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.RandomVerticalFlip(p=1.0),
                            transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.RandomVerticalFlip(p=1.0),
                            transforms.ToTensor(), norm]),
    ]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (all_probs, all_preds, all_labels)."""
    model.eval()
    all_probs = []
    all_labels = []

    for images, meta, labels in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            logits = model(images, meta)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = all_probs.argmax(axis=1)

    return all_probs, all_preds, all_labels


@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    dataset_cls,
    dataset_kwargs: dict,
    tta_transforms: list,
    device: torch.device,
    amp_dtype: torch.dtype,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TTA: average predictions over multiple augmentations."""
    model.eval()
    all_probs_sum = None
    all_labels = None

    for i, tfm in enumerate(tta_transforms):
        ds = dataset_cls(transform=tfm, **dataset_kwargs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))

        probs_list = []
        labels_list = []

        for images, meta, labels in tqdm(loader, desc=f"TTA {i+1}/{len(tta_transforms)}", leave=False):
            images = images.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits = model(images, meta)

            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(labels.numpy())

        probs = np.concatenate(probs_list)
        if all_probs_sum is None:
            all_probs_sum = probs
            all_labels = np.concatenate(labels_list)
        else:
            all_probs_sum += probs

    all_probs = all_probs_sum / len(tta_transforms)
    all_preds = all_probs.argmax(axis=1)
    return all_probs, all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ISICSkinModel")
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--tta", action="store_true", default=False,
                        help="Use test-time augmentation (4x)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # bfloat16 is preferred on all Ampere / Ada / Hopper GPUs (L40S, A100, H100)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"TTA        : {args.tta}")

    # Load model
    model = build_isic_model(num_classes=NUM_CLASSES, metadata_dim=METADATA_DIM,
                             in_chans=3, pretrained=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded epoch {ckpt['epoch']+1}, val_acc={ckpt.get('val_acc', 'N/A')}")

    # Dataset
    if args.split == "test":
        image_dir = str(DATA_DIR / "ISIC_2019_Test_Input")
        gt_csv = str(DATA_DIR / "ISIC_2019_Test_GroundTruth.csv")
        meta_csv = str(DATA_DIR / "ISIC_2019_Test_Metadata.csv")
        indices = None
    else:
        # Recreate the same val split
        from sklearn.model_selection import StratifiedShuffleSplit
        image_dir = str(DATA_DIR / "ISIC_2019_Training_Input")
        gt_csv = str(DATA_DIR / "ISIC_2019_Training_GroundTruth.csv")
        meta_csv = str(DATA_DIR / "ISIC_2019_Training_Metadata.csv")
        gt = pd.read_csv(gt_csv)
        labels_all = gt[CLASSES].values.argmax(axis=1)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        _, val_idx = next(splitter.split(np.zeros(len(labels_all)), labels_all))
        indices = val_idx

    dataset_kwargs = dict(image_dir=image_dir, gt_csv=gt_csv, meta_csv=meta_csv, indices=indices)

    if args.tta:
        tta_tfms = get_tta_transforms()
        all_probs, all_preds, all_labels = evaluate_tta(
            model, ISICDataset, dataset_kwargs, tta_tfms,
            device, amp_dtype, args.batch_size, args.workers,
        )
    else:
        ds = ISICDataset(transform=get_test_transform(), **dataset_kwargs)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))
        all_probs, all_preds, all_labels = evaluate(model, loader, device, amp_dtype)

    # Metrics
    top1 = (all_preds == all_labels).mean()
    bal = balanced_accuracy_score(all_labels, all_preds)

    # Precision / Recall / F1 (macro)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

    # AUC / AP (multiclass)
    try:
        y_true_oh = np.eye(NUM_CLASSES, dtype=np.int64)[all_labels]
        roc_auc = float(roc_auc_score(y_true_oh, all_probs, average="macro", multi_class="ovr"))
    except Exception:
        roc_auc = float("nan")

    try:
        ap = float(average_precision_score(y_true_oh, all_probs, average="macro"))
    except Exception:
        ap = float("nan")

    print(f"\n{'='*60}")
    print(f"  Results on {args.split} set ({len(all_labels)} samples)")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy    : {top1:.4f}  ({top1*100:.2f}%)")
    print(f"  Balanced Accuracy : {bal:.4f}  ({bal*100:.2f}%)")
    print(f"  Macro F1          : {f1:.4f}")
    print(f"  Macro ROC-AUC     : {roc_auc:.4f}")
    print(f"  Macro AvgPrecision: {ap:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
