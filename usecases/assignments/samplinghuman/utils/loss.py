# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

import torch


def KL_normal_loss(mean, log_var):
    mean = torch.nan_to_num(mean)
    log_var = torch.nan_to_num(log_var)
    loss = -0.5 * torch.sum((1 + log_var - mean**2 - log_var.exp()))
    loss = torch.mean(loss)
    return loss


def fft_loss(x_predicted, x_true):
    _, _, height, width = x_predicted.shape
    loss = torch.mean(
        torch.log(torch.abs(torch.fft.rfft2(x_predicted)[:, :, height // 2 : height] - torch.fft.rfft2(x_true)[:, :, width // 2 : width]) + 1)
    )
    return loss
