"""
Federated-learning utilities for structure-aware personalized aggregation.

Provides:
  - split_model_parameters : classify ISICSkinModel params into 3 groups
  - dirichlet_partition    : Dirichlet-based non-IID data splitting
  - fedavg                 : standard FedAvg aggregation
  - drift_aware_aggregate  : drift-weighted aggregation for Group B
"""

from __future__ import annotations

import copy
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn


# ====================================================================== #
#  1.  Parameter-group identification                                     #
# ====================================================================== #

# Swin stages are accessed via  model.backbone.layers[i]  in timm.
# Parameter names follow the pattern:
#   backbone.patch_embed.*        → patch embedding
#   backbone.layers.{0,1}.*      → Stage 1–2  (Group A: global)
#   backbone.layers.{2,3}.*      → Stage 3–4  (Group B: drift-aware)
#   backbone.norm.*               → final LN   (Group B)
#   metadata_mlp.*                → metadata   (Group C: local)
#   fusion_head.*                 → fusion     (Group C: local)

# Pre-compiled patterns for efficiency (no hardcoded layer names).
_PATCH_EMBED_RE = re.compile(r"^backbone\.patch_embed\.")
_STAGE_RE       = re.compile(r"^backbone\.layers\.(\d+)\.")
_NORM_RE        = re.compile(r"^backbone\.norm\.")
_LOCAL_PREFIXES = ("metadata_mlp.", "fusion_head.")

# Stage indices considered "early" (Group A) vs "deep" (Group B).
_GROUP_A_STAGES = {0, 1}
_GROUP_B_STAGES = {2, 3}


def split_model_parameters(
    model: nn.Module,
) -> Dict[str, Set[str]]:
    """
    Classify every named parameter of *model* into one of three groups.

    Returns a dict with keys ``"group_A"``, ``"group_B"``, ``"group_C"``
    mapping to sets of parameter-name strings.

    Groups
    ------
    A (global — plain FedAvg):
        patch_embed  +  backbone stages 0, 1
    B (partially global — drift-aware weighted average):
        backbone stages 2, 3  +  backbone final norm
    C (local — never aggregated):
        metadata_mlp  +  fusion_head
    """
    groups: Dict[str, Set[str]] = {"group_A": set(), "group_B": set(), "group_C": set()}

    for name, _ in model.named_parameters():
        # --- patch embedding → A ---------------------------------- #
        if _PATCH_EMBED_RE.match(name):
            groups["group_A"].add(name)
            continue

        # --- Swin stage layers ------------------------------------ #
        m = _STAGE_RE.match(name)
        if m:
            stage_idx = int(m.group(1))
            if stage_idx in _GROUP_A_STAGES:
                groups["group_A"].add(name)
            elif stage_idx in _GROUP_B_STAGES:
                groups["group_B"].add(name)
            else:
                # Safety: unexpected stage → treat as Group B
                groups["group_B"].add(name)
            continue

        # --- backbone final layer-norm → B ------------------------ #
        if _NORM_RE.match(name):
            groups["group_B"].add(name)
            continue

        # --- metadata MLP / fusion head → C ----------------------- #
        if any(name.startswith(pfx) for pfx in _LOCAL_PREFIXES):
            groups["group_C"].add(name)
            continue

        # --- Anything else → C (local, safest default) ------------ #
        groups["group_C"].add(name)

    return groups


# ====================================================================== #
#  2.  Dirichlet non-IID partition                                        #
# ====================================================================== #

def validate_partition(
    labels: np.ndarray,
    partitions: List[np.ndarray],
    min_classes: int = 2,
    min_samples: int = 100,
) -> bool:
    """
    Check that every client partition has at least *min_classes* distinct
    classes and at least *min_samples* samples.

    Returns True if valid, False otherwise.
    """
    for part in partitions:
        unique = np.unique(labels[part])
        if len(unique) < min_classes:
            return False
        if len(part) < min_samples:
            return False
    return True


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Partition dataset indices among *num_clients* via a Dirichlet distribution.

    Parameters
    ----------
    labels : 1-D int array of class labels.
    num_clients : number of federated clients.
    alpha : concentration parameter — smaller → more heterogeneous.
    seed : random seed for reproducibility.

    Returns
    -------
    List of 1-D int arrays — one per client — containing sample indices.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    N = len(labels)

    # Indices grouped by class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    # Per-client index lists (will be concatenated at the end)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = class_indices[c]
        rng.shuffle(idx_c)

        # Draw proportions from Dir(alpha, …, alpha)
        proportions = rng.dirichlet(np.full(num_clients, alpha))

        # Convert proportions to counts that sum to len(idx_c)
        counts = (proportions * len(idx_c)).astype(int)
        # Distribute remainder one-by-one
        remainder = len(idx_c) - counts.sum()
        for i in range(remainder):
            counts[i % num_clients] += 1

        start = 0
        for k in range(num_clients):
            client_indices[k].extend(idx_c[start : start + counts[k]].tolist())
            start += counts[k]

    return [np.array(ci, dtype=np.int64) for ci in client_indices]


# ====================================================================== #
#  3.  Aggregation helpers                                                #
# ====================================================================== #

def _extract_group_params(
    state_dict: OrderedDict,
    param_names: Set[str],
) -> OrderedDict:
    """Return an OrderedDict with only the keys in *param_names*."""
    return OrderedDict((k, v) for k, v in state_dict.items() if k in param_names)


def fedavg(
    client_state_dicts: Sequence[OrderedDict],
    param_names: Set[str],
    weights: Optional[Sequence[float]] = None,
) -> OrderedDict:
    """
    Standard Federated Averaging for a subset of parameters.

    Parameters
    ----------
    client_state_dicts : list of full state-dicts (one per client).
    param_names        : parameter names to aggregate.
    weights            : optional per-client weighting (e.g. by dataset size).
                         If None, uniform weighting is used.

    Returns
    -------
    OrderedDict of aggregated parameters (only the keys in *param_names*).
    """
    num_clients = len(client_state_dicts)
    if weights is None:
        w = [1.0 / num_clients] * num_clients
    else:
        total = sum(weights)
        w = [wk / total for wk in weights]

    agg: OrderedDict = OrderedDict()
    for name in param_names:
        # Weighted sum across clients — all arithmetic on the same device
        agg[name] = sum(                                      # type: ignore[assignment]
            w[k] * client_state_dicts[k][name].float()
            for k in range(num_clients)
        )
        # Cast back to original dtype
        agg[name] = agg[name].to(client_state_dicts[0][name].dtype)

    return agg


def drift_aware_aggregate(
    client_state_dicts: Sequence[OrderedDict],
    param_names: Set[str],
    epsilon: float = 1e-6,
) -> OrderedDict:
    """
    Drift-aware weighted aggregation for Group B parameters.

    For each parameter *l*:
      1. Compute the unweighted mean across clients:
            w̄_l = (1/K) Σ_k  w_{k,l}
      2. Compute drift magnitude per client:
            D_{k,l} = ‖w_{k,l} − w̄_l‖₂
      3. Inverse-drift weight:
            α_{k,l} = 1 / (D_{k,l} + ε)
      4. Normalise:
            ᾶ_{k,l} = α_{k,l} / Σ_j α_{j,l}
      5. Aggregate:
            w_l^{global} = Σ_k ᾶ_{k,l} · w_{k,l}

    Clients whose parameters drifted *less* from the mean get *more* weight.

    Parameters
    ----------
    client_state_dicts : list of full state-dicts (one per participating client).
    param_names        : parameter names belonging to Group B.
    epsilon            : small constant to avoid division by zero.

    Returns
    -------
    OrderedDict of aggregated Group B parameters.
    """
    num_clients = len(client_state_dicts)
    agg: OrderedDict = OrderedDict()

    for name in param_names:
        # Collect tensors (keep on original device, cast to float for precision)
        tensors = [client_state_dicts[k][name].float() for k in range(num_clients)]

        # Step 1: unweighted mean
        mean_t = sum(tensors) / num_clients                   # type: ignore[assignment]

        # Step 2: drift magnitudes  D_{k,l} = ‖w_{k,l} − w̄_l‖_2
        drifts = [torch.norm(t - mean_t, p=2).item() for t in tensors]

        # Step 3: inverse-drift weights  α_{k,l} = 1 / (D_{k,l} + ε)
        alphas = [1.0 / (d + epsilon) for d in drifts]

        # Step 4: normalise across clients
        alpha_sum = sum(alphas)
        norm_alphas = [a / alpha_sum for a in alphas]

        # Step 5: weighted aggregation
        agg[name] = sum(                                      # type: ignore[assignment]
            norm_alphas[k] * tensors[k] for k in range(num_clients)
        )
        # Cast back to original dtype
        agg[name] = agg[name].to(client_state_dicts[0][name].dtype)

    return agg


# ====================================================================== #
#  4.  Convenience: apply aggregated parameters to a model                #
# ====================================================================== #

def apply_params(
    model: nn.Module,
    params: OrderedDict,
) -> None:
    """Load *params* into *model* (partial update — unmatched keys untouched)."""
    current = model.state_dict()
    current.update(params)
    model.load_state_dict(current)
