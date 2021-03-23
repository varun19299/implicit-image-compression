from models.siren import Siren
import torch
from torch import nn
from torch.nn import functional as F
from utils.data import get_grid
from einops import rearrange
from pytorch_wavelets import DWTInverse, DWTForward
import kornia


class WaveletSiren(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 3,
        depth: int = 8,
        hidden_size: int = 64,
        wavelet_levels: int = 1,
        first_omega_0: float = 50.0,
        hidden_omega_0: float = 50.0,
        outermost_linear: bool = True,
        simulate_quantization: bool = False,
        small_dense_density: float = 1.0,
        **kwargs
    ):
        super().__init__()

        # Small Dense
        hidden_size = int(hidden_size * np.sqrt(small_dense_density))

        self.output_size = output_size
        self.wavelet_levels = wavelet_levels
        self.wavelet_windows = 3

        # predict Y, Cb, Cr
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

        # predict only Y
        self.HF_siren = Siren(
            input_size,
            output_size * wavelet_levels,
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

        # LF_image = rearrange(LF_image, "h w c -> 1 c h w")
        # HF_image = rearrange(
        #     HF_image,
        #     "h w (l c) -> 1 c l h w",
        #     l=self.wavelet_levels * self.wavelet_windows,
        #     c=self.output_size,
        # )
        #
        # # Inverse DWT
        # out_image = self.IDWT((LF_image, [HF_image]))

        Y_LF_image = rearrange(LF_image[:, :, 0], "h w -> 1 1 h w")
        Y_HF_image = rearrange(
            HF_image,
            "h w l -> 1 1 l h w",
            l=self.wavelet_levels * self.wavelet_windows,
        )

        # Inverse DWT
        Y_out_image = self.IDWT((Y_LF_image, [Y_HF_image]))

        # Cb, Cr
        scale = h / self.LF_h
        Cb_Cr_LF_image = rearrange(LF_image[:, :, 1:], "h w c -> 1 c h w")
        Cb_Cr_out_image = F.interpolate(
            Cb_Cr_LF_image,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )

        Y_Cb_Cr_out_image = torch.cat((Y_out_image, Cb_Cr_out_image), dim=1)

        out_image = kornia.color.ycbcr.ycbcr_to_rgb(Y_Cb_Cr_out_image)

        return rearrange(out_image, "1 c h w -> h w c")


if __name__ == "__main__":
    model = WaveletSiren()
    model(torch.rand(10, 10, 2))
