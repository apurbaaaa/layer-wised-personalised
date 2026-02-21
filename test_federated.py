"""Quick smoke-test for federated utilities."""

from collections import OrderedDict

import numpy as np
import torch

from federated_utils import (
    dirichlet_partition,
    drift_aware_aggregate,
    fedavg,
    split_model_parameters,
)
from model import build_isic_model


def test_param_groups():
    print("=== Parameter Group Identification ===")
    m = build_isic_model(pretrained=False)
    groups = split_model_parameters(m)
    all_params = set(n for n, _ in m.named_parameters())
    covered = groups["group_A"] | groups["group_B"] | groups["group_C"]
    assert covered == all_params, f"Missing: {all_params - covered}"
    print(f"  Group A: {len(groups['group_A'])} params")
    print(f"  Group B: {len(groups['group_B'])} params")
    print(f"  Group C: {len(groups['group_C'])} params")
    print(f"  Total:   {len(all_params)} params")

    for name in groups["group_A"]:
        assert "patch_embed" in name or "layers.0" in name or "layers.1" in name, name
    for name in groups["group_B"]:
        assert "layers.2" in name or "layers.3" in name or "norm" in name, name
    for name in groups["group_C"]:
        assert "metadata_mlp" in name or "fusion_head" in name, name
    print("  OK\n")


def test_dirichlet():
    print("=== Dirichlet Partition ===")
    labels = np.random.randint(0, 8, size=1000)
    parts = dirichlet_partition(labels, num_clients=5, alpha=0.5)
    assert len(parts) == 5
    total = sum(len(p) for p in parts)
    assert total == 1000
    assert len(set(np.concatenate(parts))) == 1000
    for i, p in enumerate(parts):
        dist = np.bincount(labels[p], minlength=8)
        print(f"  Client {i}: {len(p)} samples  dist={dist.tolist()}")
    print("  OK\n")


def test_fedavg():
    print("=== FedAvg ===")
    sd1 = OrderedDict({"a": torch.ones(3), "b": torch.zeros(3)})
    sd2 = OrderedDict({"a": torch.zeros(3), "b": torch.ones(3)})
    avg = fedavg([sd1, sd2], {"a", "b"})
    assert torch.allclose(avg["a"], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(avg["b"], torch.tensor([0.5, 0.5, 0.5]))
    print("  OK\n")


def test_drift_aware():
    print("=== Drift-Aware Aggregation ===")
    # Symmetric case
    sd1 = OrderedDict({"x": torch.tensor([0.1, 0.1, 0.1])})
    sd2 = OrderedDict({"x": torch.tensor([10.0, 10.0, 10.0])})
    agg = drift_aware_aggregate([sd1, sd2], {"x"})
    assert torch.allclose(agg["x"], torch.tensor([5.05, 5.05, 5.05]), atol=1e-4)
    print(f"  Symmetric: {agg['x'].tolist()}")

    # Asymmetric: 2 clients at 1.0, 1 client at 100 → result biased toward 1.0
    sd1 = OrderedDict({"x": torch.tensor([1.0, 1.0])})
    sd2 = OrderedDict({"x": torch.tensor([1.0, 1.0])})
    sd3 = OrderedDict({"x": torch.tensor([100.0, 100.0])})
    agg = drift_aware_aggregate([sd1, sd2, sd3], {"x"})
    print(f"  Asymmetric: {agg['x'].tolist()}")
    assert agg["x"][0].item() < 50.0
    print("  OK\n")


if __name__ == "__main__":
    test_param_groups()
    test_dirichlet()
    test_fedavg()
    test_drift_aware()
    print("All tests passed!")
