#!/usr/bin/env python3
"""
Shrink a federated checkpoint to everything needed for evaluation.

A full checkpoint stores the global model PLUS a complete state-dict for every
client (~2.1 GB at K=5). But evaluation only ever reconstructs a client as

    Groups A+B  <- global model
    Group C     <- that client's own head

(see evaluate_federated.py). The clients' own A/B tensors are never read: they
are the pre-aggregation local copies, overwritten at the start of every round.

So dropping them is lossless for evaluation and cuts ~2.1 GB to ~0.4 GB.
evaluate_federated.py works unchanged on the compacted file, because it only
indexes client_states for names in group_C.

What is preserved: global_model_state, each client's Group C tensors,
param_groups, args, round, metrics.
What is discarded: each client's Groups A+B (resume-from-round-25 only).

    python compact_checkpoint.py <in.pt> <out.pt>
"""
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    blob = torch.load(src, map_location="cpu", weights_only=False)

    groups = blob.get("param_groups", {})
    group_c = set(groups.get("group_C", []))
    clients = blob.get("client_states") or []

    slim_clients = []
    for cs in clients:
        slim_clients.append(OrderedDict(
            (k, v) for k, v in cs.items() if k in group_c))

    out = {
        "round": blob.get("round"),
        "global_model_state": blob["global_model_state"],
        "client_states": slim_clients,
        "param_groups": {g: list(s) for g, s in groups.items()},
        "args": blob.get("args"),
        "global_metrics": blob.get("global_metrics"),
        "client_accs": blob.get("client_accs"),
        "client_bal_accs": blob.get("client_bal_accs"),
        "best_global_bal_acc": blob.get("best_global_bal_acc"),
        "_compacted": True,
        "_note": ("client_states hold Group C only; Groups A+B come from "
                  "global_model_state, which is how evaluation reconstructs "
                  "each client."),
    }
    torch.save(out, dst)

    a, b = src.stat().st_size, dst.stat().st_size
    kept = len(slim_clients[0]) if slim_clients else 0
    print(f"{src.name}: {a/1e9:.2f} GB -> {b/1e9:.2f} GB "
          f"({100*b/a:.1f}%), {len(slim_clients)} clients x {kept} Group C tensors")


if __name__ == "__main__":
    main()
