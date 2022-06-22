import torch
import torch.nn as nn

ALPHA = .25
GAMMA = 2.0

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = torch.sigmoid(inputs)

        eps = torch.finfo(torch.float32).eps

        inputs = torch.clamp(inputs, eps, 1 - eps)

        p_t =  torch.where(targets == 1, inputs, 1 - inputs)

        alpha_factor = torch.ones_like(targets) * alpha
        alpha_t = torch.where(targets == 1, alpha_factor, 1 - alpha_factor)

        ce = -torch.log(p_t)

        weight = alpha_t * (1 - p_t)**gamma

        loss = weight * ce

        loss = torch.mean(torch.sum(loss, axis=0))
        loss = torch.sum(loss, axis=0)

        return loss
