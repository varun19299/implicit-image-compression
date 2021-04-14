import cv2
import kornia
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


class SliceDataset(Dataset):
    def __init__(
        self,
        h: int,
        w: int,
    ):
        super().__init__()
        self.h = h
        self.w = w

    def __len__(self):
        return self.h * self.w

    def __getitem__(self, index):
        h_idx = index // self.h
        w_idx = index % self.h

        return h_idx, w_idx


def get_dataloaders(img: torch.Tensor, batch_height, batch_width):
    h, w, _ = img.shape
    train_loader = DataLoader(
        SliceDataset(h, w),
        batch_size=batch_height * batch_width,
        shuffle=True,
    )

    test_loader = DataLoader(
        SliceDataset(h, w), batch_size=batch_height * batch_width, shuffle=False
    )

    return train_loader, test_loader


def load_img(
    path: str,
    height: int = 256,
    width: int = 256,
    bits: int = 8,
    plot: bool = False,
    crop_mode: str = "centre-crop",
    save_gt: bool = False,
    **kwargs
) -> torch.Tensor:
    img = cv2.imread(path, -1)[:, :, ::-1] / (2 ** bits - 1)
    img = torch.from_numpy(img.copy()).float().permute(2, 0, 1)

    if crop_mode == "resize-crop":
        # Resize such that shorter side matches corresponding target side
        smaller_side = min(height, width)
        img = kornia.resize(
            img.unsqueeze(0), smaller_side, align_corners=False
        ).squeeze(0)

    img = kornia.center_crop(img.unsqueeze(0), (height, width), align_corners=False)
    img = img.squeeze(0).permute(1, 2, 0)

    if plot:
        plt.imshow(img)
        plt.show()

    if save_gt:
        cv2.imwrite("gt.png", img.numpy()[:, :, ::-1] * 255.0)

    # H x W x 3
    return img


def get_grid(
    height: int, width: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    # Create input pixel coordinates in the unit square
    coords_h = torch.linspace(0, 1, height, device=device)
    coords_w = torch.linspace(0, 1, width, device=device)

    # H x W x 2
    grid = torch.stack(torch.meshgrid(coords_h, coords_w), dim=-1)

    return grid
