# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

import torch


def KL_normal_loss(mean, log_var):
    loss = -0.5 * torch.sum((1 + log_var - mean**2 - log_var.exp()), 1)
    loss = torch.mean(loss)
    return loss


def fft_loss(x_predicted, x_true):
    _, _, _, width = x_predicted.shape
    loss = torch.mean(torch.log(torch.abs(torch.fft.rfft2(x_predicted)[:, :, :, width // 4 :] - torch.fft.rfft2(x_true)[:, :, :, width // 4 :]) + 1))
    return loss
