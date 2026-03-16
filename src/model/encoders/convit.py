import torch
import torch.nn as nn


class GPSA(nn.Module):
    def __init__(self, d: int, nh: int, drop: float = 0.0):
        super().__init__()
        if d % nh != 0:
            raise ValueError(f"d={d} must be divisible by nh={nh}")
        self.nh = int(nh)
        self.hd = d // nh
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.gate = nn.Parameter(torch.full((nh,), -2.0))  # favor positional early
        self.pos_proj = nn.Parameter(torch.randn(nh, 3) * 0.02)
        self.register_buffer("rel", torch.empty(0), persistent=False)

    def _rel_feats(self, gh: int, gw: int, device: torch.device) -> torch.Tensor:
        n = gh * gw
        if self.rel.numel() == 0 or self.rel.size(0) != n:
            y, x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, gh, device=device),
                torch.linspace(-1.0, 1.0, gw, device=device),
                indexing="ij",
            )
            c = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)  # (n,2)
            r = c[:, None, :] - c[None, :, :]  # (n,n,2)
            d2 = (r ** 2).sum(dim=-1, keepdim=True)
            self.rel = torch.cat([r, d2], dim=-1)
        return self.rel

    def forward(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_c = (q @ k.transpose(-2, -1)) * self.scale
        attn_c = attn_c.softmax(dim=-1)
        rel = self._rel_feats(gh, gw, x.device)
        attn_p = torch.einsum("nmd,hd->hnm", rel, self.pos_proj).unsqueeze(0)
        attn_p = attn_p.softmax(dim=-1)
        g = torch.sigmoid(self.gate).view(1, self.nh, 1, 1)
        attn = (1.0 - g) * attn_p + g * attn_c
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.proj(out)
        out = self.drop(out)
        return out


class MHSA(nn.Module):
    def __init__(self, d: int, nh: int, drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return self.drop(y)


class Block(nn.Module):
    def __init__(self, d: int, nh: int, md: int, drop: float, use_gpsa: bool):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.a = GPSA(d, nh, drop) if use_gpsa else MHSA(d, nh, drop)
        self.n2 = nn.LayerNorm(d)
        self.m = nn.Sequential(
            nn.Linear(d, md),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(md, d),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        y = self.n1(x)
        y = self.a(y, gh, gw) if isinstance(self.a, GPSA) else self.a(y)
        x = x + y
        x = x + self.m(self.n2(x))
        return x


class ConViTEnc(nn.Module):
    def __init__(
        self,
        h: int = 640,
        w: int = 480,
        ps: int = 16,
        d: int = 256,
        dl: int = 6,
        nh: int = 4,
        md: int = 512,
        drop: float = 0.0,
        cls: bool = False,
        gpsa_layers: int = 2,
    ):
        super().__init__()
        if h % ps != 0 or w % ps != 0:
            raise ValueError(f"patch size {ps} must divide h={h}, w={w}")
        gh, gw = h // ps, w // ps
        self.gh = int(gh)
        self.gw = int(gw)
        self.h = int(h)
        self.w = int(w)
        self.cls = bool(cls)
        self.patch = nn.Conv2d(1, d, kernel_size=ps, stride=ps, bias=True)
        n = self.gh * self.gw
        t = n + (1 if self.cls else 0)
        self.pos = nn.Parameter(torch.randn(1, t, d) * 0.02)
        self.cls_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02) if self.cls else None
        self.drop = nn.Dropout(drop)
        blocks = []
        for i in range(dl):
            blocks.append(Block(d, nh, md, drop, use_gpsa=(i < gpsa_layers)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.size(1) == 1, f"expected (B,1,H,W), got {tuple(x.shape)}"  # shape check
        assert x.size(2) == self.h and x.size(3) == self.w, f"expected spatial {(self.h, self.w)}, got {tuple(x.shape[-2:])}"  # size check
        t = self.patch(x).flatten(2).transpose(1, 2)
        if self.cls:
            c = self.cls_tok.expand(x.size(0), -1, -1)
            t = torch.cat([c, t], dim=1)
        t = self.drop(t + self.pos)
        for b in self.blocks:
            t = b(t, self.gh, self.gw)
        t = self.norm(t)
        z = t[:, 0] if self.cls else t.mean(dim=1)
        return z
