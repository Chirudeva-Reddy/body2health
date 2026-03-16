import torch
import torch.nn as nn
from typing import Dict, Tuple, Iterable


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        # Wider stem to handle 640x480 inputs without excessive depth
        ch = [1, 32, 64, 128, 256, 256]
        self.blocks = nn.Sequential(
            ConvBlock(ch[0], ch[1]),
            ConvBlock(ch[1], ch[2]),
            ConvBlock(ch[2], ch[3]),
            ConvBlock(ch[3], ch[4]),
            ConvBlock(ch[4], ch[5]),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ch[5], 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        h = self.gap(h).flatten(1)
        z = self.proj(h)
        return z


class DualViewCNN(nn.Module):
    def __init__(self, out_meas: int = 14):
        super().__init__()
        self.front_branch = Branch()
        self.side_branch = Branch()
        self.fuse_dim = 256 * 2
        self.meas_head = nn.Linear(self.fuse_dim, out_meas)
        self.bf_head = nn.Sequential(nn.Linear(self.fuse_dim, 1), nn.Sigmoid())

    def forward(self, front_mask: torch.Tensor, side_mask: torch.Tensor, aux=None) -> Dict[str, torch.Tensor]:
        f_feat = self.front_branch(front_mask)
        s_feat = self.side_branch(side_mask)
        fs = torch.cat([f_feat, s_feat], dim=-1)
        meas = self.meas_head(fs)
        bf = self.bf_head(fs)
        return {"meas": meas, "bf": bf}

    def front_pretrain_params(self) -> Iterable[nn.Parameter]:
        return self.front_branch.parameters()

    def side_pretrain_params(self) -> Iterable[nn.Parameter]:
        return self.side_branch.parameters()

    def fusion_params(self) -> Iterable[nn.Parameter]:
        return list(self.meas_head.parameters()) + list(self.bf_head.parameters())
