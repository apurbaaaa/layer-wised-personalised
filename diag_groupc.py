#!/usr/bin/env python3
"""
Diagnostic: is the *global* model's Group C (metadata_mlp + fusion_head, i.e.
the classifier) actually updated during federated training, or does it stay at
its random initialisation?

This matters because Group C is never aggregated, so `global_model` may carry
an untrained classifier. If so, the "Global Acc" logged for decomposed arms
(full, decomp_nodrift) is not comparable to true_fedavg, whose Group C is
empty and therefore *is* aggregated.

Method: rebuild the model with the same seed the run used, so we reproduce the
exact initialisation, then compare against the checkpoint's stored global
Group C tensors.

    python diag_groupc.py <checkpoint.pt> [seed]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from federated_train import seed_everything, NUM_CLASSES, METADATA_DIM  # noqa: E402
from federated_utils import split_model_parameters  # noqa: E402
from model import build_isic_model  # noqa: E402


def main() -> None:
    ckpt_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gsd = ckpt["global_model_state"]
    stored_groups = ckpt.get("param_groups", {})
    args = ckpt.get("args", {})

    print(f"checkpoint      : {ckpt_path}")
    print(f"round           : {ckpt.get('round', -1) + 1}")
    print(f"ablation        : {args.get('ablation')}  alpha={args.get('dirichlet_alpha')}")
    print(f"group sizes     : A={len(stored_groups.get('group_A', []))} "
          f"B={len(stored_groups.get('group_B', []))} "
          f"C={len(stored_groups.get('group_C', []))}")

    # Reproduce the run's initialisation exactly.
    seed_everything(seed)
    fresh = build_isic_model(num_classes=NUM_CLASSES, metadata_dim=METADATA_DIM,
                            in_chans=3, pretrained=True)
    fsd = fresh.state_dict()
    groups = split_model_parameters(fresh)

    group_c = sorted(stored_groups.get("group_C") or groups["group_C"])
    group_a = sorted(stored_groups.get("group_A") or groups["group_A"])

    if not group_c:
        print("\nGroup C is EMPTY for this arm — every tensor is aggregated, so "
              "the global model's classifier IS trained. No confound here.")
        return

    print(f"\nComparing {len(group_c)} Group C tensors against fresh init:")
    identical = 0
    for name in group_c:
        if name not in gsd or name not in fsd:
            print(f"  [missing] {name}")
            continue
        same = torch.allclose(gsd[name].float(), fsd[name].float(), atol=1e-8)
        identical += int(same)
        if len(group_c) <= 12:
            d = (gsd[name].float() - fsd[name].float()).abs().max().item()
            print(f"  {name:35s} identical={same}  max|Δ|={d:.3e}")

    print(f"\nGroup C identical to init: {identical}/{len(group_c)}")

    # Sanity control: Group A *should* have changed (it is aggregated).
    a_same = 0
    for name in group_a[:20]:
        if name in gsd and name in fsd:
            a_same += int(torch.allclose(gsd[name].float(), fsd[name].float(), atol=1e-8))
    print(f"Group A identical to init (control, first 20): {a_same}/{min(20, len(group_a))}")

    print()
    if identical == len(group_c):
        print("VERDICT: the global model's classifier is FROZEN at random init.")
        print("  => 'Global Acc' for this arm is NOT comparable to true_fedavg,")
        print("     whose classifier is aggregated and therefore trained.")
    elif identical == 0:
        print("VERDICT: Group C in the global model IS being updated.")
        print("  => global accuracy is comparable across arms.")
    else:
        print("VERDICT: MIXED — some Group C tensors changed, some did not.")


if __name__ == "__main__":
    main()
