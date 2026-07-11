#!/usr/bin/env python3
"""Short smoke train for YOLOX MNv4-Small + PAFPN (synthetic boxes, no COCO)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch is required for smoke train") from exc


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class DwPw(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, groups=channels, bias=True)
        self.pw = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _silu(self.pw(self.dw(x)))


class DecoupledHead(nn.Module):
    def __init__(self, hidden: int, num_classes: int, num_convs: int = 2):
        super().__init__()
        self.stem = nn.Conv2d(hidden, hidden, 1, bias=True)
        self.cls_convs = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, 3, padding=1, bias=True) for _ in range(num_convs)]
        )
        self.reg_convs = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, 3, padding=1, bias=True) for _ in range(num_convs)]
        )
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1, bias=True)
        self.reg_pred = nn.Conv2d(hidden, 4, 1, bias=True)
        self.obj_pred = nn.Conv2d(hidden, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = _silu(self.stem(x))
        cls = stem
        for conv in self.cls_convs:
            cls = _silu(conv(cls))
        reg = stem
        for conv in self.reg_convs:
            reg = _silu(conv(reg))
        return torch.cat([self.reg_pred(reg), self.obj_pred(reg), self.cls_pred(cls)], dim=1)


class NanoPafpn(nn.Module):
    def __init__(self, c3: int, c4: int, c5: int, hidden: int, num_classes: int, num_convs: int = 2):
        super().__init__()
        self.lat3 = nn.Conv2d(c3, hidden, 1, bias=True)
        self.lat4 = nn.Conv2d(c4, hidden, 1, bias=True)
        self.lat5 = nn.Conv2d(c5, hidden, 1, bias=True)
        self.td_p4 = DwPw(hidden, stride=1)
        self.td_p3 = DwPw(hidden, stride=1)
        self.bu_n4 = DwPw(hidden, stride=2)
        self.bu_n5 = DwPw(hidden, stride=2)
        self.heads = nn.ModuleList(
            [DecoupledHead(hidden, num_classes, num_convs) for _ in range(3)]
        )

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        l3 = self.lat3(c3)
        l4 = self.lat4(c4)
        l5 = self.lat5(c5)
        p5 = l5
        p4 = self.td_p4(l4 + F.interpolate(p5, scale_factor=2, mode="nearest"))
        p3 = self.td_p3(l3 + F.interpolate(p4, scale_factor=2, mode="nearest"))
        n3 = p3
        n4 = self.bu_n4(n3) + p4
        n5 = self.bu_n5(n4) + p5
        outs = []
        for feat, head in zip((n3, n4, n5), self.heads):
            o = head(feat)
            outs.append(o.permute(0, 2, 3, 1).reshape(o.shape[0], -1))
        return torch.cat(outs, dim=1)


class SmokeDetector(nn.Module):
    def __init__(self, num_classes: int = 10, hidden: int = 64, freeze_backbone: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("timm is required for smoke train") from exc

        self.backbone = timm.create_model(
            "mobilenetv4_conv_small",
            pretrained=True,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # Probe channel counts
        with torch.no_grad():
            feats = self.backbone(torch.zeros(1, 3, 64, 64))
        c3, c4, c5 = (f.shape[1] for f in feats)
        self.neck = NanoPafpn(c3, c4, c5, hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4, c5 = self.backbone(x)
        return self.neck(c3, c4, c5)


def synthetic_batch(batch: int, size: int, num_classes: int, device: torch.device):
    images = torch.randn(batch, 3, size, size, device=device)
    # 1–3 random boxes per image as soft targets on the flat output (proxy loss).
    targets = torch.zeros(batch, (8 * 8 + 4 * 4 + 2 * 2) * (5 + num_classes), device=device)
    for b in range(batch):
        n = int(np.random.randint(1, 4))
        for _ in range(n):
            idx = int(np.random.randint(0, targets.shape[1] // (5 + num_classes)))
            base = idx * (5 + num_classes)
            targets[b, base : base + 4] = torch.rand(4, device=device)
            targets[b, base + 4] = 1.0
            cls = int(np.random.randint(0, num_classes))
            targets[b, base + 5 + cls] = 1.0
    return images, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "yolox_mnv4_pafpn_smoke.pt",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    model = SmokeDetector(num_classes=args.classes).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    losses = []
    model.train()
    for step in range(args.steps):
        images, targets = synthetic_batch(2, args.size, args.classes, device)
        opt.zero_grad()
        pred = model(images)
        # Align lengths if backbone feature sizes differ slightly
        n = min(pred.shape[1], targets.shape[1])
        loss = F.mse_loss(pred[:, :n], targets[:, :n])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss={losses[-1]:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "losses": losses}, args.out)
    print(f"saved {args.out}")
    if losses[0] > 0 and losses[-1] < losses[0]:
        print("loss decreased: ok")
    else:
        print("warning: loss did not clearly decrease (smoke only)")


if __name__ == "__main__":
    main()
