import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle as pkl
alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
standard_alphebet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
standard_dict = {}
for index in range(len(standard_alphebet)):
    standard_dict[standard_alphebet[index]] = index
class ImageLoss(nn.Module):

    def __init__(self, loss_weight=[1, 1e-4]):
        super(ImageLoss, self).__init__()
        self.mse = nn.MSELoss(reduce=False)
        self.GPLoss = GradientPriorLoss()
        self.loss_weight = loss_weight
        self.MaskLoss = MaskLoss()
    def forward(self, out_images, target_images):
        mse_loss = self.mse(out_images, target_images).mean(1).mean(1).mean(1)
        gp_loss = self.GPLoss(out_images[:, :3, :, :], target_images[:, :3, :, :])
        loss = self.loss_weight[0] * mse_loss + self.loss_weight[1] * gp_loss
        return loss

from torchvision import transforms
from scipy.cluster.vq import vq,kmeans
class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, hr_img, sr_mask):
        hr_mask = self.get_mask(hr_img)
        if hr_img.size(2)!=sr_mask.size(2):
            hr_mask = F.interpolate(hr_mask, size=(hr_img.shape[2]//2, hr_img.shape[3]//2), mode='bicubic', align_corners=True)
        mask_loss = self.mse_loss(sr_mask, hr_mask)
        return mask_loss

    @staticmethod
    def get_mask(img_tensor):
        b, c, h, w = img_tensor.shape
        img = img_tensor.mean(dim=1)
        toTensor = transforms.ToTensor()
        masks = torch.zeros(img.size()).to(img.device)  # 空的[B,16,64,64]
        for i in range(b):
            img_batch = img[i].reshape(-1).cpu()
            # 聚类， k=2是聚类数目
            centroids, variance = kmeans(img_batch.detach(), 2)
            code, distance = vq(img_batch.detach(), centroids)
            code = code.reshape(h, w)
            fc = sum(code[:, 0])
            lc = sum(code[:, -1])
            fr = sum(code[0, :])
            lr = sum(code[-1, :])
            num = int(fr > w // 2) + int(lr > w // 2) + int(fc > h // 2) + int(lc > h // 2)
            if num >= 3:
                mask = 1 - code
            else:
                mask = code
            mask = toTensor(mask).contiguous().to(img_tensor.device)
            masks[i:i + 1, :, :] = mask
        return masks.unsqueeze(1)


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss(reduce=False)

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)

        g_loss = self.func(map_out, map_target)

        return g_loss.mean(1).mean(1).mean(1)

    @staticmethod
    def gradient_map(x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad

def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]


class SemanticLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SemanticLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.margin = margin

        self.lambda1 = 1.0
        self.lambda2 = 1.0

        self.kl_loss = torch.nn.KLDivLoss()

    def forward(self, pred_vec, gt_vec):
        norm_vec = torch.abs(gt_vec - pred_vec)
        margin_loss = torch.mean(norm_vec)  #
        ce_loss = self.kl_loss(torch.log(pred_vec + 1e-20), gt_vec + 1e-20)
        return self.lambda1 * margin_loss + self.lambda2 * ce_loss  # ce_loss #margin_loss # + ce_loss #  + sim_loss #margin_loss +
