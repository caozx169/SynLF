import torch
import numpy as np
from torch.utils.data import Dataset
from utils.rendering import render_viewsimg
from loguru import logger
from utils.augmentation import SpecularAugmentor, TransparencyAugmentor


class MonoSyntheticDataset(Dataset):
    def __init__(self,
                 mono_dataset: str,
                 mindisp: float, maxdisp: float,
                 viewspos: list,
                 randomrotaug: bool,
                 randomdispoffset: bool,
                 randomnoise:bool,
                 randomgray:bool,
                 randomblur:bool=False,
                 randomblur_ratio: float=0.3,
                 blur_sigma_min: float=0.1,
                 blur_sigma_max: float=3.0,
                 noise_fwc: int=1e4,
                 noise_std: float=0.01,
                 pad: str = 'noise',
                 interp_method: str = 'linear',
                 add_cocblur: bool = False,
                 cocblur_binsnum: int = 11,
                 cocblur_method: str = 'naive',
                 cocblur_psf_size: int = 35,
                 cocblur_psf_scale: int = 9,
                 cocblur_num_parallel: int = 1,
                 randomocc: bool = False,
                 randomocc_ratio: float = 0.1,
                 occ_num: int = 3,
                 occ_size: tuple = (50, 50),
                 occ_color: bool = False,
                 disp_clip: bool = False,
                 random_disp_jitter: bool = False,
                 random_photometric_aug: bool = False,
                 random_photometric_aug_ratio: float = 0.3,
                 brightness: float = 0.4,
                 contrast: float = 0.4,
                 gamma: float = 0.0,
                 random_specular: bool = False,
                 random_specular_ratio: float = 0.3,
                 specular_num_shapes: list = [50, 150],
                 specular_shape_size: list = [0.05, 0.4],
                 specular_plane_prob: float = 0.5,
                 specular_pattern_prob: str = 'mixed',
                 random_transparency: bool = False,
                 random_transparency_ratio: float = 0.2,
                 transparency_alpha_range: list = [0.05, 0.25],
                 transparency_num_patches: list = [1, 3],
                 transparency_patch_size: list = [0.15, 0.4],
                 transparency_front_glass_ratio: float = 0.5,
                 transparency_texture_prob: str = 'mixed',
                 **kwargs) -> None:
        super().__init__()
        self.mono_dataset = mono_dataset
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        self.viewspos = np.array(viewspos)
        self.cropsize = mono_dataset.cropsize
        self.randomrotaug = randomrotaug
        self.randomdispoffset = randomdispoffset
        self.randomnoise = randomnoise
        self.randomblur = randomblur
        self.randomblur_ratio = randomblur_ratio
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.randomgray = randomgray
        self.randomocc = randomocc
        self.randomocc_ratio = randomocc_ratio
        self.occ_num = occ_num
        self.occ_size = occ_size
        self.occ_color = occ_color
        self.disp_clip = disp_clip
        self.add_cocblur = add_cocblur
        self.noise_fwc = noise_fwc
        self.noise_std = noise_std
        self.pad = pad
        self.interp_method = interp_method
        self.random_disp_jitter = random_disp_jitter
        self.random_photometric_aug = random_photometric_aug
        self.random_photometric_aug_ratio = random_photometric_aug_ratio
        self.random_specular = random_specular
        self.random_specular_ratio = random_specular_ratio
        self.specular_augmentor = SpecularAugmentor(
            min_disp=mindisp,
            max_disp=maxdisp,
            edge_threshold=5,
            safety_margin=15,
            num_shapes_range=specular_num_shapes,
            shape_size_range=specular_shape_size,
            plane_prob=specular_plane_prob,
        )
        self.random_transparency = random_transparency
        self.random_transparency_ratio = random_transparency_ratio
        self.transparency_augmentor = TransparencyAugmentor(
            alpha_range=transparency_alpha_range,
            min_disp=mindisp,
            max_disp=maxdisp,
            num_patches=transparency_num_patches,
            patch_size_range=transparency_patch_size,
            front_glass_ratio=transparency_front_glass_ratio,
            texture_prob=transparency_texture_prob,
        )
        logger.info(f'MonoSyntheticDataset initialized with cropsize_hw: {self.cropsize}')
        logger.info(f'Mindisp: {self.mindisp}, Maxdisp: {self.maxdisp}')
        logger.info(f'Viewspos: {self.viewspos}')
        logger.info(f'randomrotaug: {self.randomrotaug}, randomdispoffset: {self.randomdispoffset}, randomnoise: {self.randomnoise}, noise_std: {self.noise_std}')
        logger.info(f'noise_fwc: {self.noise_fwc}, noise_std: {self.noise_std}')
        logger.info(f'randomocc: {self.randomocc}, randomocc_ratio: {self.randomocc_ratio}, occ_num: {self.occ_num}, occ_size: {self.occ_size}, occ_color: {self.occ_color}')
        logger.info(f'random_disp_jitter: {self.random_disp_jitter}, disp_clip: {self.disp_clip}')
        logger.info(f'randomblur: {self.randomblur}, randomblur_ratio: {self.randomblur_ratio}, blur_sigma_min: {self.blur_sigma_min}, blur_sigma_max: {self.blur_sigma_max}')
        logger.info(f'random_photometric_aug: {self.random_photometric_aug}, random_photometric_aug_ratio: {self.random_photometric_aug_ratio}, brightness: {brightness}, contrast: {contrast}, gamma: {gamma}')
        logger.info(f'random_specular: {self.random_specular}, ratio: {self.random_specular_ratio}, num_shapes: {specular_num_shapes}, size: {specular_shape_size}, plane_prob: {specular_plane_prob}')
        logger.info(f'random_transparency: {self.random_transparency}, ratio: {self.random_transparency_ratio}, front_glass_ratio: {transparency_front_glass_ratio}, texture_prob: {transparency_texture_prob}')
        if self.add_cocblur:
            self.cocblur_binsnum = cocblur_binsnum
            self.cocblur_method = cocblur_method
            logger.info(f'add_cocblur: {self.add_cocblur}, cocblur_binsnum: {self.cocblur_binsnum}, cocblur_method: {self.cocblur_method}, psf_size: {cocblur_psf_size}, psf_scale: {cocblur_psf_scale}, num_parallel: {cocblur_num_parallel}')

    def randomrotation(self, img, disparity, mask, viewspos):
        p = np.random.rand()
        if p > 0.333:
            # 随机执行horizontal flip或vertical flip
            flip_dim = np.random.randint(2)
            img = np.flip(img, flip_dim)
            disparity = np.flip(disparity, flip_dim)
            mask = np.flip(mask, flip_dim)
        p = np.random.rand()
        if p > 0.25:
            k = np.random.randint(1, 4)  # 随机选择旋转90°、180°或270°
            img = np.rot90(img, k=k, axes=(0, 1))
            disparity = np.rot90(disparity, k=k, axes=(0, 1))
            mask = np.rot90(mask, k=k, axes=(0, 1))
            if k == 1:
                viewspos[:, 1] = - viewspos[:, 1]
                viewspos = viewspos[:, [1, 0]]
            elif k == 2:
                viewspos[:, 0] = - viewspos[:, 0]
                viewspos[:, 1] = - viewspos[:, 1]
            elif k == 3:
                viewspos[:, 0] = - viewspos[:, 0]
                viewspos = viewspos[:, [1, 0]]
        return img, disparity, mask, viewspos

    def set_cropsize(self, cropsize):
        self.mono_dataset.set_cropsize(cropsize)
        self.cropsize = cropsize
        logger.info(f'set cropsize_hw to {cropsize}')

    def __len__(self):
        return len(self.mono_dataset)


    def _pre_augment(self, lambertian, disparity, finite_mask, viewspos):
        if self.randomrotaug:
            lambertian, disparity, finite_mask, viewspos = self.randomrotation(
                lambertian, disparity, finite_mask, viewspos
            )
        depth = 1 + (disparity.max() - disparity)
        if self.randomdispoffset:
            p90 = np.percentile(disparity, 90)
            if p90 < self.maxdisp:
                disparity = disparity + np.random.uniform(0, self.maxdisp - p90)
        if self.disp_clip:
            disparity = np.clip(disparity, self.mindisp, self.maxdisp)
        return lambertian, disparity, finite_mask, viewspos, depth



    def __getitem__(self, index):
        _, lambertian, disparity = self.mono_dataset[index]
        finite_mask = np.isfinite(disparity)
        disparity = np.where(finite_mask, disparity, self.mindisp - 0.1)
        # lambertian: (H, W, C) int8, range [0, 255]
        lambertian, disparity, finite_mask, viewspos, depth = self._pre_augment(
            lambertian, disparity, finite_mask, self.viewspos.copy()
        )
        dispforrender = disparity
        if self.random_specular and np.random.rand() < self.random_specular_ratio:
            dispforrender = self.specular_augmentor(disparity, lambertian)

        # 用 dispforrender 重新计算 depth，保证 z-buffer 遮挡关系与视差一致
        # （Specular 区域视差变小→更远→depth应更大；若用原始 depth 则遮挡关系错误）
        depth_for_render = 1 + (dispforrender.max() - dispforrender)

        viewsimg, viewsimg_masks, views_disp = render_viewsimg(lambertian, depth_for_render, dispforrender, viewspos, pad=self.pad, interp_method=self.interp_method,
        random_occ=self.randomocc, random_occ_ratio=self.randomocc_ratio, occ_num=self.occ_num, occ_size=self.occ_size, occ_color=self.occ_color,
        random_disp_jitter=self.random_disp_jitter)
        if self.random_transparency and np.random.rand() < self.random_transparency_ratio:
            viewsimg, disparity = self.transparency_augmentor(
                viewsimg, disparity, depth, viewspos, self.pad, self.interp_method)
        disparity_mask = finite_mask & (disparity > self.mindisp) & (disparity < self.maxdisp)
        # import tifffile as tiff
        viewsimg = viewsimg.astype(np.float32) / 255.0
        viewsimg = viewsimg.transpose(0, 3, 1, 2)
        return {
            'viewimgs': torch.from_numpy(viewsimg).float(),
            'views_disp': torch.from_numpy(views_disp.copy()).float(),
            'disp': torch.from_numpy(disparity.copy()).float()[None],
            'masks': torch.from_numpy(disparity_mask)[None],
            'viewspos': torch.from_numpy(viewspos),
            'viewsimg_masks': torch.from_numpy(viewsimg_masks).bool(),
        }
