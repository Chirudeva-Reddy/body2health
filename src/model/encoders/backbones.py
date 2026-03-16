import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18, resnet34
except Exception:  # torchvision may be missing in lightweight envs
    resnet18 = None
    resnet34 = None

try:
    from torchvision.models import convnextv2_tiny
except Exception:  # optional in older torchvision versions
    convnextv2_tiny = None


def _mk_model(fn, pt: bool) -> nn.Module:
    if not pt:
        try:
            return fn(weights=None)
        except TypeError:
            return fn(pretrained=False)
    try:
        return fn(weights="DEFAULT")
    except Exception:
        try:
            return fn(weights=None, pretrained=True)
        except Exception:
            return fn(pretrained=True)


class _TVEnc(nn.Module):
    def __init__(self, m: nn.Module, fd: int, d: int):
        super().__init__()
        self.m = m
        self.a = nn.Identity() if fd == d else nn.Linear(fd, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.size(1) in {1, 3}
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # keep pretrained RGB filters for contours
        z = self.m(x)
        if z.ndim != 2:
            z = torch.flatten(z, 1)
        return self.a(z)


def mk_enc(name: str, d: int, pt: bool) -> nn.Module:
    if name == "resnet18":
        if resnet18 is None:
            raise ImportError("torchvision.resnet18 is unavailable")
        m = _mk_model(resnet18, pt)
        m.fc = nn.Identity()
        return _TVEnc(m, fd=512, d=d)
    if name == "resnet34":
        if resnet34 is None:
            raise ImportError("torchvision.resnet34 is unavailable")
        m = _mk_model(resnet34, pt)
        m.fc = nn.Identity()
        return _TVEnc(m, fd=512, d=d)
    if name == "convnextv2_tiny":
        if convnextv2_tiny is None:
            raise ValueError("convnextv2_tiny not available in this torchvision build")
        m = _mk_model(convnextv2_tiny, pt)
        if hasattr(m, "classifier"):
            m.classifier = nn.Identity()
        return _TVEnc(m, fd=768, d=d)
    raise ValueError(f"unsupported backbone: {name}")
