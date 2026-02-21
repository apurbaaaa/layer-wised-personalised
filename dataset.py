"""
ISIC 2019 Dataset — PyTorch Dataset for skin lesion classification.

Produces:
  image  : Tensor [3, 384, 384]
  meta   : Tensor [13]   (1 age + 3 sex + 9 site)
  label  : int in [0..7]
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ------------------------------------------------------------------ #
#  Fixed category mappings (deterministic across federated clients)   #
# ------------------------------------------------------------------ #
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

SEX_CATS = ["female", "male", "unknown"]           # 3

SITE_CATS = [
    "anterior torso",
    "head/neck",
    "lateral torso",
    "lower extremity",
    "oral/genital",
    "palms/soles",
    "posterior torso",
    "upper extremity",
    "unknown",                                      # 9th = missing
]

AGE_MAX = 85.0  # normalisation ceiling


# ------------------------------------------------------------------ #
#  Helper: encode metadata row → numpy float32[13]                   #
# ------------------------------------------------------------------ #
def encode_metadata(
    age: float,
    sex: str,
    site: str,
) -> np.ndarray:
    """Return a float32 vector of length 13."""
    vec = np.zeros(13, dtype=np.float32)

    # 1) age — normalised [0, 1], NaN → 0
    if np.isfinite(age):
        vec[0] = age / AGE_MAX
    # else: already 0.0

    # 2) sex one-hot  (indices 1, 2, 3)
    sex_str = str(sex).strip().lower() if pd.notna(sex) else "unknown"
    if sex_str not in SEX_CATS:
        sex_str = "unknown"
    vec[1 + SEX_CATS.index(sex_str)] = 1.0

    # 3) anatom site one-hot  (indices 4 .. 12)
    site_str = str(site).strip().lower() if pd.notna(site) else "unknown"
    if site_str not in SITE_CATS:
        site_str = "unknown"
    vec[4 + SITE_CATS.index(site_str)] = 1.0

    return vec


# ------------------------------------------------------------------ #
#  Dataset                                                            #
# ------------------------------------------------------------------ #
class ISICDataset(Dataset):
    """
    ISIC 2019 dataset.

    Args:
        image_dir:   Path to folder containing ISIC_*.jpg files.
        gt_csv:      Path to ground-truth CSV (one-hot columns for each class).
        meta_csv:    Path to metadata CSV (age_approx, sex, anatom_site_general).
        transform:   torchvision / albumentations transform for images.
        indices:      Optional subset of row indices (for train/val split).
    """

    def __init__(
        self,
        image_dir: str,
        gt_csv: str,
        meta_csv: str,
        transform: Optional[Callable] = None,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        gt_df = pd.read_csv(gt_csv)
        meta_df = pd.read_csv(meta_csv)

        # Merge on image id
        df = gt_df.merge(meta_df, on="image", how="inner")

        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.image_dir = image_dir
        self.transform = transform

        # Pre-compute labels (argmax of one-hot)
        self.image_ids: list[str] = df["image"].tolist()
        self.labels: np.ndarray = df[CLASSES].values.argmax(axis=1).astype(np.int64)

        # Pre-compute metadata vectors
        self.metadata: np.ndarray = np.stack([
            encode_metadata(
                row["age_approx"],
                row.get("sex", np.nan),
                row.get("anatom_site_general", np.nan),
            )
            for _, row in df.iterrows()
        ])  # [N, 13]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = os.path.join(self.image_dir, f"{self.image_ids[idx]}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        meta = torch.from_numpy(self.metadata[idx])  # float32[13]
        label = int(self.labels[idx])

        return image, meta, label


# ------------------------------------------------------------------ #
#  Class weights for imbalanced sampling / loss                      #
# ------------------------------------------------------------------ #
def compute_class_weights(gt_csv: str) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    gt = pd.read_csv(gt_csv)
    counts = gt[CLASSES].sum().values.astype(np.float64)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(CLASSES)
    return torch.tensor(weights, dtype=torch.float32)
