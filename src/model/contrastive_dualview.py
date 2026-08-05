import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Tuple
from src.model.dualview import Branch
from .bbox_features import extract_dual_bbox_features
from .encoders.backbones import mk_enc


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual and (in_ch == out_ch and stride == 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            identity = x if self.downsample is None else self.downsample(x)
            out = out + identity
        return out


class LargeBranch(nn.Module):
    def __init__(self, base_dim: int = 64):
        super().__init__()
        # Wider and deeper: 1 -> 64 -> 128 -> 256 -> 512 -> 512
        ch = [1, base_dim, base_dim*2, base_dim*4, base_dim*8, base_dim*8]
        self.stem = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            ConvBlock(ch[1], ch[2], stride=1, use_residual=True),
            ConvBlock(ch[2], ch[2], stride=1, use_residual=True),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(ch[2], ch[3], stride=2, use_residual=True),
            ConvBlock(ch[3], ch[3], stride=1, use_residual=True),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(ch[3], ch[4], stride=2, use_residual=True),
            ConvBlock(ch[4], ch[4], stride=1, use_residual=True),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(ch[4], ch[5], stride=2, use_residual=True),
            ConvBlock(ch[5], ch[5], stride=1, use_residual=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ch[5], 512)  # larger embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        z = self.proj(x)
        return z

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualViewContrastive(nn.Module):
    """
    Dual-branch encoder with:
    - Projection heads for InfoNCE/NT-Xent contrastive loss.
    - Regression head for tape-measured body dimensions.
    Optional: concatenate bbox features for robustness to blob silhouettes.
    """

    def __init__(
        self,
        out_meas: int = 2,
        proj_dim: int = 128,
        use_large: bool = False,
        base_dim: int = 64,
        use_bbox_features: bool = False,
        encoder: str = "cnn",
        enc: str | None = None,
        pt: bool = False,
        convit_patch_size: int = 16,
        convit_dim: int = 256,
        convit_depth: int = 6,
        convit_heads: int = 4,
        convit_mlp_dim: int = 512,
        convit_drop: float = 0.0,
        convit_pool: str = "mean",
        convit_shared: bool = True,
        convit_gpsa_layers: int = 2,
        convit_img_hw: Tuple[int, int] = (640, 480),
    ):
        """
        Args:
            out_meas: number of dimension targets (e.g., waist_cm + hip_cm = 2).
            proj_dim: embedding dimension for contrastive loss.
            use_large: if True, use larger residual branch; else use original Branch.
            base_dim: base channel width for large branch.
            use_bbox_features: if True, concatenate bbox features before regression.
        """
        super().__init__()
        encoder = enc or encoder
        if encoder not in {"cnn", "resnet18", "resnet34", "convnextv2_tiny"}:
            raise ValueError(f"unsupported encoder: {encoder}")
        branch_cls = LargeBranch if use_large else Branch
        target_dim = 512 if use_large else 256
        self.use_large = use_large
        self.use_bbox_features = use_bbox_features
        self.encoder = encoder

        if encoder == "cnn":
            self.front_branch = branch_cls(base_dim=base_dim) if use_large else Branch()
            self.side_branch = branch_cls(base_dim=base_dim) if use_large else Branch()
            embed_dim = self.front_branch.proj.out_features
            self.front_adapter = nn.Identity()
            self.side_adapter = nn.Identity()
        else:
            # Pretrained CNNs give stable low-level contour filters for silhouettes.
            self.front_branch = mk_enc(encoder, d=target_dim, pt=pt)
            self.side_branch = self.front_branch  # shared view encoder
            self.front_adapter = nn.Identity()
            self.side_adapter = nn.Identity()
            embed_dim = target_dim

        self.fuse_dim = embed_dim * 2
        self.proj_f = ProjectionHead(embed_dim, proj_dim)
        self.proj_s = ProjectionHead(embed_dim, proj_dim)
        # Determine regression input dimension
        reg_in_dim = self.fuse_dim + (8 if use_bbox_features else 0)
        if use_large:
            self.meas_head = nn.Sequential(
                nn.Linear(reg_in_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, out_meas),
            )
        else:
            self.meas_head = nn.Linear(reg_in_dim, out_meas)

    def _encode(self, front_mask: torch.Tensor, side_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_feat = self.front_adapter(self.front_branch(front_mask))
        s_feat = self.side_adapter(self.side_branch(side_mask))
        return f_feat, s_feat

    def encode_views(self, front_mask: torch.Tensor, side_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns normalized embeddings (B, proj_dim) for front and side.
        """
        f_feat, s_feat = self._encode(front_mask, side_mask)
        f_z = F.normalize(self.proj_f(f_feat), dim=-1)
        s_z = F.normalize(self.proj_s(s_feat), dim=-1)
        return f_z, s_z

    def forward(self, front_mask: torch.Tensor, side_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_feat, s_feat = self._encode(front_mask, side_mask)
        fs = torch.cat([f_feat, s_feat], dim=-1)
        # Optionally concatenate bbox features
        if self.use_bbox_features:
            bbox_feat = extract_dual_bbox_features(front_mask, side_mask)  # (B, 8)
            fs = torch.cat([fs, bbox_feat], dim=-1)
        meas = self.meas_head(fs)
        f_z = F.normalize(self.proj_f(f_feat), dim=-1)
        s_z = F.normalize(self.proj_s(s_feat), dim=-1)
        return {"meas": meas, "f_z": f_z, "s_z": s_z}

    def front_pretrain_params(self) -> Iterable[nn.Parameter]:
        return list(self.front_branch.parameters()) + list(self.front_adapter.parameters())

    def side_pretrain_params(self) -> Iterable[nn.Parameter]:
        return list(self.side_branch.parameters()) + list(self.side_adapter.parameters())

    def fusion_params(self) -> Iterable[nn.Parameter]:
        return list(self.meas_head.parameters()) + list(self.proj_f.parameters()) + list(self.proj_s.parameters())
