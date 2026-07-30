#!/usr/bin/env python3
"""
Evaluate a federated checkpoint on the validation and/or test split, using the
CORRECT classifier for each arm.

Why this exists
---------------
Group C (metadata_mlp + fusion_head) is the entire classifier and is never
aggregated, so a decomposed run's ``global_model_state`` keeps its RANDOM
initial classifier for the whole run (verified by diag_groupc.py). Evaluating
that global state directly — as ``evaluate.py`` does — measures an untrained
head and is meaningless for those arms.

This script instead evaluates, for every client k:

    Groups A+B  <- aggregated global model
    Group C     <- client k's own trained head  (from ``client_states``)

and reports per-client metrics plus their mean. For an arm with an empty
Group C (``true_fedavg``) every client is identical, so a single evaluation is
performed and reported as the shared model — which makes the mean per-client
accuracy directly comparable across all arms.

Usage
-----
    python evaluate_federated.py --checkpoint <ckpt.pt> --split val
    python evaluate_federated.py --checkpoint <ckpt.pt> --split test
    python evaluate_federated.py --checkpoint <ckpt.pt> --split both --json out.json
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from dataset import CLASSES, ISICDataset
from evaluate import evaluate as run_forward
from evaluate import get_test_transform
from model import build_isic_model

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
NUM_CLASSES = 8
METADATA_DIM = 13
VAL_FRAC = 0.15
SEED = 42


def build_loader(split: str, batch_size: int, workers: int,
                 device: torch.device, limit: int = 0) -> DataLoader:
    if split == "test":
        image_dir = str(DATA_DIR / "ISIC_2019_Test_Input")
        gt_csv = str(DATA_DIR / "ISIC_2019_Test_GroundTruth.csv")
        meta_csv = str(DATA_DIR / "ISIC_2019_Test_Metadata.csv")
        indices = None
    else:
        image_dir = str(DATA_DIR / "ISIC_2019_Training_Input")
        gt_csv = str(DATA_DIR / "ISIC_2019_Training_GroundTruth.csv")
        meta_csv = str(DATA_DIR / "ISIC_2019_Training_Metadata.csv")
        gt = pd.read_csv(gt_csv)
        labels_all = gt[CLASSES].values.argmax(axis=1)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC,
                                          random_state=SEED)
        _, val_idx = next(splitter.split(np.zeros(len(labels_all)), labels_all))
        indices = val_idx

    ds = ISICDataset(image_dir=image_dir, gt_csv=gt_csv, meta_csv=meta_csv,
                     indices=indices, transform=get_test_transform())
    if limit:
        # Smoke-test path: exercise the whole code path on a few samples so a
        # broken checkpoint/dataset is caught without a full GPU pass.
        from torch.utils.data import Subset
        ds = Subset(ds, range(min(limit, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=(device.type == "cuda"))


def metrics_from(preds: np.ndarray, labels: np.ndarray,
                 probs: np.ndarray) -> Dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    try:
        y_oh = np.eye(NUM_CLASSES, dtype=np.int64)[labels]
        auc = float(roc_auc_score(y_oh, probs, average="macro", multi_class="ovr"))
    except Exception:
        auc = float("nan")
    return {
        "acc": float((preds == labels).mean()),
        "bal_acc": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1),
        "macro_auc": auc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="val",
                    choices=["val", "test", "both"])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", type=str, default=None,
                    help="Write results to this JSON path")
    ap.add_argument("--limit", type=int, default=0,
                    help="Smoke test: evaluate only the first N samples")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gsd: OrderedDict = ckpt["global_model_state"]
    client_states: List[OrderedDict] = ckpt.get("client_states") or []
    groups = ckpt.get("param_groups", {})
    group_c = set(groups.get("group_C", []))
    cargs = ckpt.get("args", {})

    logger.info("checkpoint : %s", args.checkpoint)
    logger.info("round      : %s", ckpt.get("round", -1) + 1)
    logger.info("ablation   : %s  alpha=%s  K=%s", cargs.get("ablation"),
                cargs.get("dirichlet_alpha"), cargs.get("num_clients"))
    logger.info("Group C    : %d tensors %s", len(group_c),
                "(empty -> single shared model)" if not group_c else "(personalized)")

    model = build_isic_model(num_classes=NUM_CLASSES, metadata_dim=METADATA_DIM,
                            in_chans=3, pretrained=False).to(device)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    out: Dict[str, dict] = {
        "checkpoint": args.checkpoint,
        "round": ckpt.get("round", -1) + 1,
        "ablation": cargs.get("ablation"),
        "dirichlet_alpha": cargs.get("dirichlet_alpha"),
        "num_clients": cargs.get("num_clients"),
        "group_c_size": len(group_c),
        "splits": {},
    }

    for split in splits:
        loader = build_loader(split, args.batch_size, args.workers, device,
                              limit=args.limit)
        logger.info("--- %s split: %d samples ---", split, len(loader.dataset))

        if not group_c or not client_states:
            # Single shared model (true_fedavg): one evaluation suffices.
            model.load_state_dict(gsd)
            probs, preds, labels = run_forward(model, loader, device, amp_dtype)
            m = metrics_from(preds, labels, probs)
            logger.info("shared model: acc=%.4f bal_acc=%.4f f1=%.4f auc=%.4f",
                        m["acc"], m["bal_acc"], m["macro_f1"], m["macro_auc"])
            out["splits"][split] = {
                "mode": "shared",
                "per_client": [m],
                "mean_acc": m["acc"],
                "mean_bal_acc": m["bal_acc"],
                "worst_acc": m["acc"],
                "std_acc": 0.0,
                "confusion": confusion_matrix(labels, preds).tolist(),
            }
        else:
            per_client = []
            last_cm = None
            for k, cs in enumerate(client_states):
                sd = OrderedDict()
                for name in gsd:
                    sd[name] = cs[name] if name in group_c else gsd[name]
                model.load_state_dict(sd)
                probs, preds, labels = run_forward(model, loader, device, amp_dtype)
                m = metrics_from(preds, labels, probs)
                per_client.append(m)
                last_cm = confusion_matrix(labels, preds)
                logger.info("client %d: acc=%.4f bal_acc=%.4f f1=%.4f auc=%.4f",
                            k, m["acc"], m["bal_acc"], m["macro_f1"], m["macro_auc"])

            accs = [m["acc"] for m in per_client]
            bals = [m["bal_acc"] for m in per_client]
            out["splits"][split] = {
                "mode": "personalized",
                "per_client": per_client,
                "mean_acc": float(np.mean(accs)),
                "mean_bal_acc": float(np.mean(bals)),
                "worst_acc": float(np.min(accs)),
                "worst_bal_acc": float(np.min(bals)),
                "std_acc": float(np.std(accs)),
                "std_bal_acc": float(np.std(bals)),
                "confusion": last_cm.tolist() if last_cm is not None else None,
            }
            s = out["splits"][split]
            logger.info("MEAN acc=%.4f bal_acc=%.4f | WORST acc=%.4f | STD acc=%.4f",
                        s["mean_acc"], s["mean_bal_acc"], s["worst_acc"], s["std_acc"])

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        logger.info("wrote %s", args.json)


if __name__ == "__main__":
    main()
