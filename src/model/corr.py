import torch
import torch.nn.functional as F


def bilinear_sampler(img, coords):
    width = img.shape[-1]
    xgrid = 2 * coords[..., 0:1] / (width - 1) - 1
    ygrid = coords[..., 1:2]
    grid = torch.cat([xgrid, ygrid], dim=-1).to(img.dtype)
    return F.grid_sample(img, grid, align_corners=True)


class GeoEncodingVolume:
    """Sample local multi-scale features from a disparity-indexed 3D volume."""

    def __init__(self, geo_volume, num_levels=2, dx=None):
        self.num_levels = num_levels
        self.dx = dx
        batch, channels, disparity, height, width = geo_volume.shape
        self.volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, channels, 1, disparity
        ).contiguous()

    def __call__(self, disp, coords=None, low_memory=False):
        del coords, low_memory
        batch, _, height, width = disp.shape
        dx = self.dx.to(disp.device)
        disp_flat = disp.reshape(batch * height * width, 1, 1, 1)
        dx_base = dx.reshape(1, 1, dx.numel(), 1)
        levels = torch.arange(self.num_levels, device=disp.device).view(1, self.num_levels, 1, 1)
        x_coords = disp_flat + dx_base * (2**levels)
        all_coords = torch.cat([x_coords, torch.zeros_like(x_coords)], dim=-1)
        sampled = bilinear_sampler(self.volume, all_coords)
        output = sampled.transpose(1, 2).reshape(batch, height, width, -1)
        return output.permute(0, 3, 1, 2).contiguous()
