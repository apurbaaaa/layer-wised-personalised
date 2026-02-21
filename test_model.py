"""
Smoke-test for ISICSkinModel.

Validates:
  1. 3-channel forward pass  (RGB only)
  2. 4-channel forward pass  (RGB + mask)
  3. AMP autocast compatibility
  4. Backward pass / gradient flow
  5. DDP wrapping (CPU-only, single process)
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(__file__))

from model import ISICSkinModel, build_isic_model


def _header(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def test_forward_3ch() -> None:
    """Standard 3-channel RGB forward pass."""
    _header("Test 1 — 3-channel forward pass")

    model = build_isic_model(in_chans=3, pretrained=False)
    model.eval()

    B = 2
    images = torch.randn(B, 3, 384, 384)
    metadata = torch.randn(B, 13)

    with torch.no_grad():
        logits = model(images, metadata)

    print(f"  Input  images : {images.shape}")
    print(f"  Input  meta   : {metadata.shape}")
    print(f"  Output logits : {logits.shape}")
    assert logits.shape == (B, 8), f"Expected (2, 8), got {logits.shape}"
    print("  ✓ PASSED")


def test_forward_4ch() -> None:
    """4-channel (RGB + mask) forward pass with patch-embed surgery."""
    _header("Test 2 — 4-channel forward pass")

    model = build_isic_model(in_chans=4, pretrained=False)
    model.eval()

    B = 2
    images = torch.randn(B, 4, 384, 384)
    metadata = torch.randn(B, 13)

    with torch.no_grad():
        logits = model(images, metadata)

    print(f"  Input  images : {images.shape}")
    print(f"  Output logits : {logits.shape}")
    assert logits.shape == (B, 8)
    print("  ✓ PASSED")


def test_amp() -> None:
    """AMP autocast on CPU (bfloat16)."""
    _header("Test 3 — AMP autocast (CPU bfloat16)")

    model = build_isic_model(in_chans=3, pretrained=False)
    model.eval()

    B = 2
    images = torch.randn(B, 3, 384, 384)
    metadata = torch.randn(B, 13)

    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
        logits = model(images, metadata)

    print(f"  Logits dtype: {logits.dtype}")
    assert logits.shape == (B, 8)
    print("  ✓ PASSED")


def test_backward() -> None:
    """Verify gradient flow through all branches."""
    _header("Test 4 — Backward pass / gradient flow")

    model = build_isic_model(in_chans=3, pretrained=False)
    model.train()

    B = 2
    images = torch.randn(B, 3, 384, 384)
    metadata = torch.randn(B, 13)

    logits = model(images, metadata)
    loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 8, (B,)))
    loss.backward()

    # Check gradients exist in each sub-module
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    print(f"  Loss: {loss.item():.4f}")
    print("  All parameters received gradients.")
    print("  ✓ PASSED")


def test_ddp_wrapping() -> None:
    """Wrap model with DDP on CPU (gloo backend, single rank)."""
    _header("Test 5 — DistributedDataParallel wrapping (CPU/gloo)")

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Initialise a trivial single-process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )

    model = build_isic_model(in_chans=3, pretrained=False)
    ddp_model = DDP(model)

    B = 2
    images = torch.randn(B, 3, 384, 384)
    metadata = torch.randn(B, 13)

    logits = ddp_model(images, metadata)
    loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 8, (B,)))
    loss.backward()

    print(f"  DDP logits shape: {logits.shape}")
    print(f"  DDP loss: {loss.item():.4f}")
    print("  ✓ PASSED")

    dist.destroy_process_group()


def main() -> None:
    print("=" * 60)
    print("  ISICSkinModel — Smoke Tests")
    print("=" * 60)

    test_forward_3ch()
    test_forward_4ch()
    test_amp()
    test_backward()
    test_ddp_wrapping()

    _header("ALL TESTS PASSED ✅")


if __name__ == "__main__":
    main()
