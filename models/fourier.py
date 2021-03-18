import torch
from torch import nn
import numpy as np

from einops import rearrange


class Encoding(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        map_size: int = 256,
        map_scale: float = 10.0,
    ):
        super().__init__()

        self.B = nn.Parameter(
            torch.randn(input_size, map_size) * map_scale, requires_grad=False
        )

    def forward(self, x: torch.Tensor):
        xp = (2 * np.pi * x) @ self.B
        encoding_ll = [torch.sin(xp), torch.cos(xp)]
        return torch.cat(encoding_ll, dim=-1)


class FourierNet(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 3,
        depth: int = 8,
        hidden_size: int = 256,
        map_size: int = 256,
        map_scale: float = 10.0,
        **kwargs,
    ):
        super().__init__()

        # mapping to hidden, first layer
        layers = [nn.Linear(map_size * 2, hidden_size)]
        layers.append(nn.ReLU(inplace=True))

        # hidden layers
        for _ in range(depth - 3):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))

        # output layer
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())

        self.encoding = Encoding(input_size, map_size, map_scale)
        self.layers = nn.Sequential(*layers)

    def forward(self, grid: torch.Tensor):
        # Flatten grid
        x = rearrange(grid, "h w c -> (h w) c")

        x = self.encoding(x)

        h, w, _ = grid.shape
        return rearrange(x, "(h w) c -> h w c", h=h, w=w)
