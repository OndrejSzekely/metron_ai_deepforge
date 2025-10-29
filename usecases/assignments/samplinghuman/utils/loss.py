# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

import torch


def KL_normal_loss(mean, log_var):
    loss = -0.5 * torch.sum((1 + log_var - mean**2 - log_var.exp()), 1)
    loss = torch.mean(loss)
    return loss


def fft_loss(x_predicted, x_true):
    _, _, _, width = x_predicted.shape
    loss = torch.mean(torch.log(torch.abs(torch.fft.rfft2(x_predicted)[:, :, :, : width // 2] - torch.fft.rfft2(x_true)[:, :, :, : width // 2]) + 1))
    return loss


def grouping_loss(x_predicted, y_true, mixed_images_num):
    batch_size = x_predicted.shape[0]
    x_predicted = torch.nn.functional.softmax(x_predicted, dim=-1)
    x_predicted = torch.argmax(x_predicted, dim=-1)
    res = x_predicted == y_true
    res = res.view(batch_size, mixed_images_num, -1).all(dim=-1).sum(dim=-1) / mixed_images_num

    return res.mean()
