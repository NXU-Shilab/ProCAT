"""Model definitions — depends only on PyTorch."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float):
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.fc2(self.drop(self.act(self.fc1(self.norm(x))))))
        return x + h


class ResMLPEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_blocks: int,
        expansion: int, dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, expansion, dropout) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x)


class CosineClassifier(nn.Module):
    def __init__(
        self, hidden_dim: int, num_classes: int, scale: float, learnable_scale: bool,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer("scale", torch.tensor(scale))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return (
            F.normalize(feat, dim=-1)
            @ F.normalize(self.weight, dim=-1).t()
            * self.scale
        )


class Model(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, args):
        super().__init__()
        self.encoder = ResMLPEncoder(
            input_dim, args.hidden_dim, args.resmlp_blocks,
            args.resmlp_expansion, args.dropout,
        )
        self.classifier = CosineClassifier(
            args.hidden_dim, num_classes, args.cosine_scale, args.learnable_scale,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        return feat, self.classifier(feat)
