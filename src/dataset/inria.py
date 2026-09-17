from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.image_io import imread


class InriaLFDataset(Dataset):
    """Inria DLFD benchmark reader for VisDepth evaluation."""

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
            raise FileNotFoundError(f"Inria scenes not found under {self.root}: {missing}")
        if not self.scene_names:
            raise RuntimeError(f"No Inria scenes found under {self.root}")

        grid = torch.stack(torch.meshgrid(torch.arange(9), torch.arange(9), indexing="ij"), dim=-1)
        flat_positions = (grid - torch.tensor([4, 4])).reshape(-1, 2)
        self.view_indices = [
            int(torch.where((flat_positions == position).all(dim=1))[0].item())
            for position in self.viewspos
        ]

    def __getitem__(self, index):
        name = self.scene_names[index]
        scene = self.root / name
        image_paths = [scene / f"lf_{view_id // 9 + 1}_{view_id % 9 + 1}.png" for view_id in self.view_indices]
        images = [imread(path, False, self.image_mode) for path in image_paths]
        viewimgs = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        disparity = torch.from_numpy(np.load(scene / "disparity_5_5.npy")).float().unsqueeze(0)
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
