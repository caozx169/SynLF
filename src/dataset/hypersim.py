import os
import cv2
import h5py
import numpy as np
import json
import re

from torch.utils.data import Dataset
from loguru import logger

from utils.depth_ops import compute_lf_disparity
from utils.image_io import imread, get_all_files
from utils.transforms import randomcrop, normalize_cropsize

def distance2depth(distance, focal, width, height):
    npyImageplaneX = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([height, width, 1], focal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = distance / np.linalg.norm(npyImageplane, 2, 2) * focal
    return npyDepth

class Hypersim(Dataset):
    """Hypersim Dataset，使用 compute_lf_disparity 统一映射 depth → LF disparity。"""

    # Hypersim 深度中位数的分段阈值。
    MEDIAN_THRESHOLDS = [1, 10, 100]

    def __init__(self, rootpath: str, cropsize, stats_path=None, exclusion_list=None,
                 randomresize=False, resize_factor=(0.8,1.2),
                 randomdispscale=False, dispscale_factor=(0.5,2),
                 randomblock=False, randomgray=False, image_mode='L',
                 median_multipliers=(16, 8, 4, 2), beta=-9.0, **kwargs) -> None:
        super().__init__()
        file_path = exclusion_list or os.path.join(rootpath, 'file_list.txt')
        stat_json_path = stats_path or os.path.join(rootpath, 'hypersim_robust_stats.json')
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                file_list = [file.strip() for file in f if file.strip()]
        else:
            file_list = []
            logger.warning(f'Exclusion list not found. Tried: {file_path}')
        self.file_list = file_list
        rootpath = os.path.join(rootpath, 'hypersim')
        self.rootpath = rootpath
        # Relative exclusion entries start at the scene directory (ai_xxx_xxx).
        excluded_paths = {
            os.path.normpath(os.path.join(self.rootpath, file))
            for file in self.file_list
        }
        self.randomdispscale = randomdispscale
        self.dispscale_factor = dispscale_factor
        self.focal = 0.5 * 1024 / np.tan(1/6 * np.pi)
        self.randomgray = randomgray
        self.image_mode = image_mode
        self.median_multipliers = list(median_multipliers)
        self.beta = beta

        self.depth_stats = {}

        if os.path.exists(stat_json_path):
            with open(stat_json_path, 'r') as f:
                self.depth_stats = json.load(f)
            logger.info(f'Loaded depth statistics from {stat_json_path}, {len(self.depth_stats)} scenes')
        else:
            logger.warning(f'Depth statistics file not found. Tried: {stat_json_path}')
        self.tonemap_files = get_all_files(self.rootpath, '.tonemap.jpg')
        self.tonemap_files = [file for file in self.tonemap_files
                              if os.path.normpath(file) not in excluded_paths]
        self.lambertian_files = [file.replace('tonemap.jpg', 'lambertian.jpg') for file in self.tonemap_files]
        self.residual_files = [file.replace('lambertian.jpg', 'residual.jpg') for file in self.lambertian_files]
        self.depth_files = [file.replace('lambertian.jpg', 'depth_meters.hdf5').replace('final_preview', 'geometry_hdf5') for file in self.lambertian_files]
        self.lambertian_hdf5_files = [file.replace('lambertian.jpg', 'diffuse_lambertian.hdf5').replace('final_preview', 'final_hdf5') for file in self.lambertian_files]
        self.residual_hdf5_files = [file.replace('lambertian.jpg', 'residual.hdf5').replace('final_preview', 'final_hdf5') for file in self.lambertian_files]

        # 提取每个文件对应的场景ID（格式：ai_xxx_xxx）
        self.scene_ids = []
        for lambertian_file in self.lambertian_files:
            match = re.search(r'ai_\d+_\d+', lambertian_file)
            if match:
                scene_id = match.group(0)
            else:
                scene_id = None
            self.scene_ids.append(scene_id)

        self.randomresize = randomresize
        self.resize_factor = resize_factor
        self.randomblock = randomblock
        self.imgsize_hw = imread(self.tonemap_files[0], False, self.image_mode).shape[:2]
        self.cropsize, self.randomcrop = normalize_cropsize(cropsize, self.imgsize_hw)

        self.sample_num = 1
        logger.info(f'Hypersim initialized with cropsize_hw: {self.cropsize}')
        logger.info(f'Number of files: {len(self.lambertian_files)}')
        logger.info(f'Sample number: {self.sample_num}')
        logger.info(f'randomresize: {self.randomresize}, resize_factor: {self.resize_factor}')
        logger.info(f'randomdispscale: {self.randomdispscale}, dispscale_factor: {self.dispscale_factor}')
        logger.info(f'randomblock: {self.randomblock}')
        logger.info(f'randomgray: {self.randomgray}')
        logger.info(f'image_mode: {self.image_mode}')
        logger.info(f'median_multipliers: {self.median_multipliers}, beta: {self.beta}')

    def set_cropsize(self, cropsize):
        self.cropsize = cropsize
        self.sample_num = 1
        logger.info(f'set cropsize_hw to {cropsize}')
        logger.info(f'Sample number: {self.sample_num}')

    def __len__(self) -> int:
        return len(self.lambertian_files) * self.sample_num

    def __getitem__(self, index):
        index = index // self.sample_num
        img = imread(self.tonemap_files[index], False, self.image_mode)

        distance = h5py.File(self.depth_files[index], 'r')['dataset'][:][None].astype(np.float32)
        depth = distance2depth(distance, self.focal, 1024, 768)[0]
        valid_mask = (depth > 0.1) & (np.isfinite(depth))
        depth[~valid_mask] = 0

        scene_id = self.scene_ids[index]
        if scene_id and scene_id in self.depth_stats:
            stats = self.depth_stats[scene_id]
            disparity = compute_lf_disparity(
                depth, valid_mask, "depth", stats,
                self.MEDIAN_THRESHOLDS, self.median_multipliers,
                beta=self.beta, p_scale=0.9
            )
        else:
            if scene_id:
                logger.warning(f'No statistics found for scene {scene_id}, using fallback')
            disparity = 20 / depth + self.beta
            disparity[~valid_mask] = self.beta

        if self.randomresize and self.randomcrop:
            new_H = np.random.randint(max(self.cropsize[0] + 1, int(self.resize_factor[0]*self.imgsize_hw[0])), int(self.resize_factor[1]*self.imgsize_hw[0]))
            new_W = new_H / self.imgsize_hw[0] * self.imgsize_hw[1] if np.random.rand() < 0.8 else np.random.randint(self.cropsize[1] + 1, 1.5*self.imgsize_hw[1] + 1)
            new_H = int(((new_H) // 32) * 32)
            new_W = int(((new_W) // 32) * 32)
            img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
            disparity = cv2.resize(disparity, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        if self.randomdispscale:
            disparity = disparity * np.random.uniform(self.dispscale_factor[0], self.dispscale_factor[1])
        if self.randomcrop:
            pad_mask = np.ones_like(disparity)
            img, disparity, pad_mask = randomcrop(self.cropsize, img, disparity, pad_mask)
            disparity = np.where(pad_mask > 0.5, disparity, np.nan)  # padding 区域置为 nan
        return self.lambertian_files[index], img.reshape(self.cropsize[0], self.cropsize[1], -1), disparity
