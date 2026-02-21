"""
ISICSkinModel — SwinV2-Base + Metadata MLP for ISIC Skin Lesion Classification.

Designed for federated learning (ADMM) experiments.
Supports:
  - 3-channel (RGB) or 4-channel (RGB + mask) image inputs
  - AMP (torch.amp / torch.cuda.amp)
  - Gradient accumulation (no internal normalization)
  - DistributedDataParallel (no hardcoded device, all buffers on model device)
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ISICSkinModel(nn.Module):
    """
    Multi-modal classifier for ISIC skin lesion images.

    Args:
        num_classes:      Number of output classes (default 8 for ISIC).
        metadata_dim:     Dimensionality of tabular metadata input (default 13).
        in_chans:         Number of input image channels.  Use 4 when a
                          segmentation mask is concatenated to the RGB image.
        drop_rate:        Dropout probability in the fusion / metadata heads.
        pretrained:       Whether to load ImageNet-pretrained backbone weights.
    """

    BACKBONE_NAME = "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"

    def __init__(
        self,
        num_classes: int = 8,
        metadata_dim: int = 13,
        in_chans: int = 3,
        drop_rate: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        #  1. Backbone — SwinV2-Base (classifier head removed)               #
        # ------------------------------------------------------------------ #
        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,          # removes the classifier head
            in_chans=3,             # always load with 3-chan weights first
        )

        # Infer the backbone feature dimension automatically
        backbone_dim: int = self.backbone.num_features
        print(f"[ISICSkinModel] Backbone feature dim: {backbone_dim}")

        # ------------------------------------------------------------------ #
        #  2. Optionally adapt patch_embed to 4-channel input                #
        # ------------------------------------------------------------------ #
        if in_chans == 4:
            self._adapt_patch_embed_to_4ch()

        # ------------------------------------------------------------------ #
        #  3. Metadata branch — 2-layer MLP                                  #
        # ------------------------------------------------------------------ #
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # ------------------------------------------------------------------ #
        #  4. Fusion head                                                    #
        # ------------------------------------------------------------------ #
        self.fusion_head = nn.Sequential(
            nn.Linear(backbone_dim + 128, 512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

        # Store for introspection
        self.backbone_dim = backbone_dim
        self.num_classes = num_classes
        self.in_chans = in_chans

    # ------------------------------------------------------------------ #
    #  Patch-embed surgery for 4-channel inputs                           #
    # ------------------------------------------------------------------ #
    def _adapt_patch_embed_to_4ch(self) -> None:
        """
        Replace the first convolution in ``patch_embed.proj`` so that it
        accepts 4 input channels instead of 3.  The original 3-channel
        pretrained weights are preserved; the 4th channel is initialised
        to zero so the model starts from the pretrained manifold.
        """
        old_proj: nn.Conv2d = self.backbone.patch_embed.proj  # type: ignore[assignment]

        new_proj = nn.Conv2d(
            in_channels=4,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )

        # Copy pretrained weights for the first 3 channels
        with torch.no_grad():
            new_proj.weight[:, :3, :, :] = old_proj.weight
            new_proj.weight[:, 3:, :, :] = 0.0  # zero-init 4th channel
            if old_proj.bias is not None and new_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        self.backbone.patch_embed.proj = new_proj
        print("[ISICSkinModel] Adapted patch_embed.proj to 4-channel input.")

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: torch.Tensor,    # [B, 3 or 4, 384, 384]
        metadata: torch.Tensor,  # [B, 13]
    ) -> torch.Tensor:           # [B, num_classes]
        """
        Full forward pass: backbone → metadata MLP → fusion head.

        Compatible with ``torch.amp.autocast`` and gradient-accumulation
        loops (no internal loss scaling or normalisation).
        """
        img_features = self.backbone(images)          # [B, backbone_dim]
        meta_features = self.metadata_mlp(metadata)   # [B, 128]
        fused = torch.cat([img_features, meta_features], dim=1)
        logits = self.fusion_head(fused)              # [B, num_classes]
        return logits


# ---------------------------------------------------------------------- #
#  Convenience factory                                                    #
# ---------------------------------------------------------------------- #
def build_isic_model(
    num_classes: int = 8,
    metadata_dim: int = 13,
    in_chans: int = 3,
    drop_rate: float = 0.3,
    pretrained: bool = True,
) -> ISICSkinModel:
    """Instantiate and return an ISICSkinModel."""
    model = ISICSkinModel(
        num_classes=num_classes,
        metadata_dim=metadata_dim,
        in_chans=in_chans,
        drop_rate=drop_rate,
        pretrained=pretrained,
    )
    return model
