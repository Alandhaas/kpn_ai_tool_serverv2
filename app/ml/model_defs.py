import torch
import torch.nn.functional as F
from torch import nn


class ConvNeXtFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.stem = backbone.stem
        self.stages = backbone.stages

    def forward(self, x):
        x = self.stem(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        f2 = self.stages[2](x)
        f3 = self.stages[3](f2)
        return f2, f3


class ConceptHead(nn.Module):
    def __init__(self, n_concepts: int = 4):
        super().__init__()
        self.reduce2 = nn.Conv2d(512, 576, 1)
        self.reduce3 = nn.Conv2d(1024, 448, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Linear(256, n_concepts)

    def forward(self, feats, return_fused: bool = False):
        f2, f3 = feats
        x = torch.cat(
            [
                self.reduce2(f2),
                F.interpolate(
                    self.reduce3(f3),
                    size=f2.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
            ],
            dim=1,
        )

        x = self.fuse(x)

        if return_fused:
            x.retain_grad()

        pooled = self.pool(x).flatten(1)
        logits = self.mlp(pooled)

        if return_fused:
            return logits, x
        return logits


class ThresholdAwareFinalHead(nn.Module):
    def __init__(self, thr_logit: torch.Tensor):
        super().__init__()
        K = thr_logit.numel()
        self.register_buffer("thr_logit", thr_logit)
        self.w = nn.Parameter(torch.ones(K))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(8.0))
        self.delta = nn.Parameter(torch.full((K,), 0.5))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        d = logits - self.thr_logit
        neg = F.relu(-d)
        gate = torch.sigmoid(self.alpha * (torch.abs(d) - self.delta))
        penalty = (neg * gate * self.w).sum(dim=1)
        return self.b - penalty
