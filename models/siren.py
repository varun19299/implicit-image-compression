import torch
from torch import nn
import numpy as np


class SineLayer(nn.Module):
    """
    Paper: https://arxiv.org/abs/2006.09661

    Source: https://github.com/vsitzmann/siren
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    non-linearity. Different signals may require different omega_0 in the first layer - this is a
    hyper-parameter.

    If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x: torch.Tensor):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    def __init__(
        self,
        depth: int = 4,
        input_size: int = 2,
        output_size: int = 3,
        hidden_size: int = 256,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()

        layers = [
            SineLayer(input_size, hidden_size, is_first=True, omega_0=first_omega_0)
        ]

        for _ in range(depth):
            layers.append(
                SineLayer(
                    hidden_size, hidden_size, is_first=False, omega_0=hidden_omega_0
                )
            )

        layers.append(
            SineLayer(hidden_size, output_size, is_first=False, omega_0=hidden_omega_0)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = (x - 0.5) * 2
        return self.layers(x) / 2 + 0.5
