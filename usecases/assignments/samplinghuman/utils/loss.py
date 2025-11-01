# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

import torch


def KL_normal_loss(mean, log_var):
    loss = -0.5 * torch.sum((1 + log_var - mean**2 - log_var.exp()), 1)
    loss = torch.mean(loss)
    return loss


def fft_loss(x_predicted, x_true):
    _, _, _, width = x_predicted.shape
    loss = torch.mean(torch.log(torch.abs(torch.fft.rfft2(x_predicted)[:, :, :, width // 3 :] - torch.fft.rfft2(x_true)[:, :, :, width // 3 :]) + 1))
    return loss


def grouping_loss(x_predicted, y_true, mixed_images_num):
    batch_size = x_predicted.shape[0]
    x_predicted = torch.nn.functional.softmax(x_predicted, dim=-1)
    x_predicted = torch.argmax(x_predicted, dim=-1)
    res = x_predicted == y_true
    res = res.view(batch_size, mixed_images_num, -1).all(dim=-1).sum(dim=-1) / mixed_images_num

    return res.mean()


def focal_loss(x_predicted, y_true, alpha=0.75, gamma=2):
    # x_predicted (B, fragments_num, groups_num)
    # y_true (B, fragments_num)
    x_predicted = x_predicted.transpose(1, 2)
    y_true_mat = torch.zeros_like(x_predicted, dtype=torch.int32)
    y_true_mat.scatter_(2, y_true.unsqueeze(-1), 1)

    loss = -alpha * (1 - x_predicted) ** gamma * torch.log(x_predicted.clamp(min=1e-8)) * y_true_mat
    loss += -(1 - alpha) * x_predicted**gamma * torch.log((1 - x_predicted).clamp(min=1e-8)) * (1 - y_true_mat)
    loss = loss.sum(dim=(-1, -2))
    return loss.mean()


def centroid_loss(x_predicted, x_encoded, y, groups_num, embeddings_num):
    centers = torch.zeros((x_predicted.size(0), groups_num, embeddings_num), dtype=torch.float32, device="cuda")
    for sample_ind in range(y.size(0)):
        centers[sample_ind].index_add_(0, y[sample_ind], x_encoded[sample_ind])
    centers = centers / (x_encoded.size(1) / groups_num)
    return torch.nn.functional.mse_loss(x_predicted, centers)
