import configparser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.image_io import imread, read_pfm


class HCI_LFDepthDataset(Dataset):
    """Four-dimensional HCI benchmark reader for VisDepth evaluation."""

    def __init__(
        self,
        rootpath,
        mindisp,
        maxdisp,
        viewspos,
        image_mode="L",
        scenes=None,
        **unused,
    ):
        del unused
        self.root = Path(rootpath)
        self.mindisp = float(mindisp)
        self.maxdisp = float(maxdisp)
        self.viewspos = torch.tensor(viewspos, dtype=torch.float32)
        self.image_mode = image_mode

        available = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.scene_names = list(scenes) if scenes is not None else available
        missing = sorted(set(self.scene_names) - set(available))
        if missing:
            raise FileNotFoundError(f"HCI scenes not found under {self.root}: {missing}")
        if not self.scene_names:
            raise RuntimeError(f"No HCI scenes found under {self.root}")

        config = configparser.ConfigParser()
        config.read(self.root / self.scene_names[0] / "parameters.cfg")
        rows = int(config.get("extrinsics", "num_cams_x"))
        cols = int(config.get("extrinsics", "num_cams_y"))
        center = torch.tensor([(rows - 1) // 2, (cols - 1) // 2])
        grid = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij"), dim=-1)
        flat_positions = (grid - center).reshape(-1, 2)
        self.view_indices = [
            int(torch.where((flat_positions == position).all(dim=1))[0].item())
            for position in self.viewspos
        ]

    def __getitem__(self, index):
        name = self.scene_names[index]
        scene = self.root / name
        images = [imread(scene / f"input_Cam{view_id:03d}.png", False, self.image_mode) for view_id in self.view_indices]
        viewimgs = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        disparity, _ = read_pfm(scene / "gt_disp_lowres.pfm")
        disparity = torch.from_numpy(disparity.copy()).float().unsqueeze(0)
        mask = torch.isfinite(disparity) & (disparity >= self.mindisp) & (disparity <= self.maxdisp)
        disparity = torch.where(mask, disparity, torch.zeros_like(disparity))
        return {
            "name": name,
            "viewimgs": viewimgs,
            "disp": disparity,
            "masks": mask,
            "viewspos": self.viewspos,
        }

    def __len__(self):
        return len(self.scene_names)
