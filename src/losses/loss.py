import torch
import torch.nn as nn
import torch.nn.functional as F


def ssi_normalize(x):
    # x: BxCxHxW
    flatten_x = x.flatten(-2)
    median = flatten_x.median(dim=-1, keepdim=True)[0]
    mad = torch.abs(flatten_x - median).mean(dim=-1, keepdim=True)
    return (x - median[..., None]) / (mad[..., None] + 1e-8)


class MultiScaleNormLoss(nn.Module):
    def __init__(self, scale=4, p_norm=1, trim=0):
        super(MultiScaleNormLoss, self).__init__()
        self.scales_num = scale
        self.p_norm = p_norm
        self.trim = trim
    def forward(self, pred, gt, mask, ssi=False):
        loss = 0.0
        pred_scale = pred.clone()
        gt_scale = gt.clone()
        mask_scale = mask.clone()
        for i in range(self.scales_num):
            if i > 0:
                pred_scale = F.interpolate(pred_scale, scale_factor=0.5, mode='bilinear', align_corners=True)
                gt_scale = F.interpolate(gt_scale, scale_factor=0.5, mode='bilinear', align_corners=True)
                mask_scale = F.interpolate(mask_scale.float(), scale_factor=0.5, mode='bilinear', align_corners=True) > 0.99
            if ssi:
                pred_scale = ssi_normalize(pred_scale)
                gt_scale = ssi_normalize(gt_scale)
            scale_loss = torch.pow(torch.abs(pred_scale - gt_scale), self.p_norm)[mask_scale]
            if self.trim > 0:
                trim_num = int(self.trim * scale_loss.shape[0])
                scale_loss = scale_loss.sort()[0][:-trim_num].sum() / (mask_scale.sum() - trim_num + 1e-8)
            else:
                scale_loss = scale_loss.sum() / (mask_scale.sum() + 1e-8)
            loss += scale_loss
        loss = loss / self.scales_num
        return loss


class MultiScaleDerivativeLoss(nn.Module):
    def __init__(self, scale=4, p_norm=1, order=1, ssi=False, trim=0):
        super(MultiScaleDerivativeLoss, self).__init__()
        self.scales_num = scale
        self.p_norm = p_norm
        self.ssi = ssi
        self.trim = trim
        if order == 1:
            kernal_x = torch.tensor([[ -3, 0,  3],
                           [-10, 0, 10],
                           [ -3, 0,  3]]).float().view(1,1,3,3)
            kernal_y = torch.tensor([[ -3, -10, -3],
                           [ 0,  0,  0],
                           [ 3,  10,  3]]).float().view(1,1,3,3)
            self.register_buffer('kernel', torch.cat([kernal_x, kernal_y], dim=0))
        elif order == 2:
            self.register_buffer('kernel', torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]]).float().view(1,1,3,3))

    def forward(self, pred, gt, mask):
        loss = 0.0
        pred_scale = pred.clone()
        gt_scale = gt.clone()
        mask_scale = mask.clone()
        for i in range(self.scales_num):
            if i > 0:
                pred_scale = F.interpolate(pred_scale, scale_factor=0.5, mode='bilinear', align_corners=True)
                gt_scale = F.interpolate(gt_scale, scale_factor=0.5, mode='bilinear', align_corners=True)
                mask_scale = F.interpolate(mask_scale.float(), scale_factor=0.5, mode='bilinear', align_corners=True) > 0.99
            if self.ssi:
                pred_scale = ssi_normalize(pred_scale)
                gt_scale = ssi_normalize(gt_scale)
            pred_deravative = F.conv2d(pred_scale, self.kernel, padding=0)
            gt_deravative = F.conv2d(gt_scale, self.kernel, padding=0)
            mask_scale_pad = mask_scale[..., 1:-1, 1:-1].repeat(1, pred_deravative.shape[1], 1, 1)
            scale_loss = torch.pow(torch.abs(pred_deravative - gt_deravative), self.p_norm)[mask_scale_pad]
            if self.trim > 0:
                trim_num = int(self.trim * scale_loss.shape[0])
                scale_loss = scale_loss.sort()[0][:-trim_num].sum() / (mask_scale_pad.sum() - trim_num + 1e-8)
            else:
                scale_loss = scale_loss.sum() / (mask_scale_pad.sum() + 1e-8)
            loss += scale_loss
        loss = loss / self.scales_num
        return loss



class SeqLoss(nn.Module):
    def __init__(self, gamma=0.9, base_loss=MultiScaleDerivativeLoss(scale=4, p_norm=1, order=1)):
        super(SeqLoss, self).__init__()
        self.gamma = gamma
        self.base_loss = base_loss
    def forward(self, preds, gt, mask):
        factor = self.gamma ** (1 + torch.arange(len(preds), device=preds[0].device))
        factor = factor.flip(0)
        loss = 0.0
        for i in range(len(preds)):
            loss += self.base_loss(preds[i], gt, mask) * factor[i]
        return loss