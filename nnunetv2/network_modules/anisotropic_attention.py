import torch
from torch import nn


class AnisotropicAttention(nn.Module):
    """
    Direction-aware attention that suppresses streak artifacts from limited-angle
    back-projection by computing features along each anatomical axis separately.

    Three parallel branches use anisotropic 3D convolutions:
      - Axial:   (1, 3, 3) kernel — sensitive to in-plane (Z-slice) artifacts
      - Sagittal: (3, 1, 3) kernel — sensitive to Y-slice artifacts
      - Coronal:  (3, 3, 1) kernel — sensitive to X-slice artifacts

    The three feature maps are fused via channel attention (SE block) and gated
    through a sigmoid to produce a soft attention map. Residual connection ensures
    the module can degrade to identity if the attention is not useful.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()

        self.conv_axial = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3),
                                    padding=(0, 1, 1))
        self.conv_sag = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 3),
                                  padding=(1, 0, 1))
        self.conv_cor = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1),
                                  padding=(1, 1, 0))

        self.norm_axial = nn.InstanceNorm3d(in_channels)
        self.norm_sag = nn.InstanceNorm3d(in_channels)
        self.norm_cor = nn.InstanceNorm3d(in_channels)

        fusion_in = in_channels * 3
        self.fusion = nn.Conv3d(fusion_in, in_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.act = nn.LeakyReLU(inplace=True)

        reduced = max(fusion_in // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(fusion_in, reduced, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(reduced, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.gate = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Initialize gate bias so sigmoid ~ 0 -> near-identity at start
        nn.init.constant_(self.gate[0].bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_axial = self.norm_axial(self.conv_axial(x))
        f_sag = self.norm_sag(self.conv_sag(x))
        f_cor = self.norm_cor(self.conv_cor(x))

        concat = torch.cat([f_axial, f_sag, f_cor], dim=1)
        se_weight = self.se(concat)

        fused = self.fusion(concat)
        fused = fused * se_weight
        fused = self.norm(fused)
        fused = self.act(fused)

        attn = self.gate(fused)
        return x * attn + x
