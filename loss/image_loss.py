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

    def forward(self, out_images, target_images):
        loss = (self.loss_weight[0] * self.mse(out_images, target_images).mean(1).mean(1).mean(1) + \
                self.loss_weight[1] * self.GPLoss(out_images[:, :3, :, :], target_images[:, :3, :, :]))
        return loss

class GradientPriorLoss(nn.Module):
    def __init__(self):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss()

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        return self.func(map_out, map_target)

    @staticmethod
    def gradient_map(x):
        _, _, h_x, w_x = x.size()

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
