from models.siren import Siren
import torch
from torch import nn
from utils.data import get_grid
from einops import rearrange
from pytorch_wavelets import DWTInverse, DWTForward


class WaveletSiren(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 3,
        depth: int = 8,
        hidden_size: int = 256,
        wavelet_levels: int = 1,
        first_omega_0: float = 50.0,
        hidden_omega_0: float = 50.0,
        outermost_linear: bool = True,
        simulate_quantization: bool = False,
        **kwargs
    ):
        super().__init__()
        self.output_size = output_size
        self.wavelet_levels = wavelet_levels
        self.wavelet_windows = 3

        self.LF_siren = Siren(
            input_size,
            output_size,
            depth,
            hidden_size,
            first_omega_0,
            hidden_omega_0,
            outermost_linear,
            simulate_quantization,
            **kwargs
        )

        self.HF_siren = Siren(
            input_size,
            output_size * wavelet_levels * 3,
            depth,
            hidden_size,
            first_omega_0,
            hidden_omega_0,
            outermost_linear,
            simulate_quantization,
            **kwargs
        )

        self.IDWT = DWTInverse(mode="zero", wave="db3")
        self.DWT = DWTForward(J=self.wavelet_levels, wave="db3", mode="zero")
        self.LF_h = None

    def forward(self, grid: torch.Tensor):
        device = grid.device
        h, w, _ = grid.shape

        if not self.LF_h:
            Yl, Yh_ll = self.DWT(torch.rand(1, 1, h, w).to(device))
            _, _, self.LF_h, self.LF_w = Yl.shape
            self.HF_h_ll = [Yh.shape[-2] for Yh in Yh_ll]
            self.HF_w_ll = [Yh.shape[-1] for Yh in Yh_ll]

        LF_grid = get_grid(self.LF_h, self.LF_w, device=device)
        LF_image = self.LF_siren(LF_grid)  # H x W x 3

        HF_grid = get_grid(self.HF_h_ll[0], self.HF_h_ll[0], device=device)
        HF_image = self.HF_siren(HF_grid)  # H x W x 3 * levels

        LF_image = rearrange(LF_image, "h w c -> 1 c h w")
        HF_image = rearrange(
            HF_image,
            "h w (l c) -> 1 c l h w",
            l=self.wavelet_levels * self.wavelet_windows,
            c=self.output_size,
        )

        # Inverse DWT
        out_image = self.IDWT((LF_image, [HF_image]))
        return out_image


if __name__ == "__main__":
    model = WaveletSiren()
    model(torch.rand(10, 10, 2))
