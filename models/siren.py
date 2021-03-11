import torch
from torch import nn
import numpy as np

from dataclasses import dataclass


@dataclass(eq=False, repr=False)
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

    in_features: int
    out_features: int
    bias: bool = True
    is_first: bool = False
    omega_0: float = 30.0
    no_activation: bool = False
    simulate_quantization: bool = False

    def __post_init__(self):
        super().__init__()
        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            bound = 1 / self.in_features
        else:
            bound = np.sqrt(6 / self.in_features) / self.omega_0
        self.linear.weight.uniform_(-bound, bound)
        setattr(self.linear, "scaler", bound)

    def forward(self, x: torch.Tensor):
        if self.simulate_quantization:
            x = self.quant(x)

        x = self.linear(x) * self.omega_0

        if self.simulate_quantization:
            x = self.dequant(x)
        if not self.no_activation:
            x = torch.sin(x)
        return x


class Siren(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 3,
        depth: int = 8,
        hidden_size: int = 256,
        first_omega_0: float = 50.0,
        hidden_omega_0: float = 50.0,
        outermost_linear: bool = True,
        simulate_quantization: bool = False,
        **kwargs
    ):
        super().__init__()

        layers = [
            SineLayer(
                input_size,
                hidden_size,
                is_first=True,
                omega_0=first_omega_0,
                simulate_quantization=simulate_quantization,
            )
        ]

        for _ in range(depth - 2):
            layers.append(
                SineLayer(
                    hidden_size,
                    hidden_size,
                    omega_0=hidden_omega_0,
                    simulate_quantization=simulate_quantization,
                )
            )

        layers.append(
            SineLayer(
                hidden_size,
                output_size,
                omega_0=hidden_omega_0,
                no_activation=outermost_linear,
                simulate_quantization=simulate_quantization,
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # 0...1 -> -1...1
        x = (x - 0.5) * 2
        # -1...1 -> 0...1
        return self.layers(x) / 2 + 0.5
