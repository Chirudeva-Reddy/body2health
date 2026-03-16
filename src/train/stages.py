import torch
import torch.nn as nn
from typing import Iterator, Tuple


def _fake_loader(batch_size: int = 8, steps: int = 100) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    H, W = 256, 128
    for _ in range(steps):
        front = (torch.rand(batch_size, 1, H, W) > 0.5).float()
        side  = (torch.rand(batch_size, 1, H, W) > 0.5).float()
        y_meas = torch.rand(batch_size, 14)
        y_bf   = torch.rand(batch_size, 1)
        yield front, side, y_meas, y_bf


def _debug_shapes(front, side, out):
    print("front:", tuple(front.shape), "side:", tuple(side.shape))
    print("meas:", tuple(out["meas"].shape), "bf:", tuple(out["bf"].shape))


def train_front(model: nn.Module, steps: int = 100) -> nn.Module:
    model.train()
    opt = torch.optim.Adam(model.front_pretrain_params(), lr=1e-3)
    H, W = 256, 128
    for front, side, y_meas, y_bf in _fake_loader(batch_size=8, steps=steps):
        front = front
        side = side
        y_meas = y_meas
        with torch.no_grad():
            side.zero_()
        out = model(front, side)
        if (out["meas"].ndim != 2) or (out["bf"].ndim != 2):
            _debug_shapes(front, side, out)
        loss = (out["meas"] - y_meas).abs().mean() + (out["bf"] - y_bf).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def train_side(model: nn.Module, steps: int = 100) -> nn.Module:
    model.train()
    opt = torch.optim.Adam(model.side_pretrain_params(), lr=1e-3)
    for front, side, y_meas, y_bf in _fake_loader(batch_size=8, steps=steps):
        with torch.no_grad():
            front.zero_()
        out = model(front, side)
        if (out["meas"].ndim != 2) or (out["bf"].ndim != 2):
            _debug_shapes(front, side, out)
        loss = (out["meas"] - y_meas).abs().mean() + (out["bf"] - y_bf).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def finetune_fusion(model: nn.Module, steps: int = 50) -> nn.Module:
    model.train()
    opt = torch.optim.Adam(model.fusion_params(), lr=3e-4)
    for front, side, y_meas, y_bf in _fake_loader(batch_size=8, steps=steps):
        out = model(front, side)
        if (out["meas"].ndim != 2) or (out["bf"].ndim != 2):
            _debug_shapes(front, side, out)
        loss = (out["meas"] - y_meas).abs().mean() + (out["bf"] - y_bf).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
