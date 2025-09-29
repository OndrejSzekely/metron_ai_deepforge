# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""DiffusionDet: Diffusion Model for Object Detection (https://arxiv.org/abs/2211.09788)"""

from torch import Tensor, nn


class DiffusionDet(nn.Module):
    """DiffusionDet model class.

    Attributes:
        image_encoder(nn.Module): Image encoder module.
        detection_decoder(nn.Module): RoI feature extractor module.
        detection_head(nn.Module): Detection head module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Identity()
        self.detection_decoder = nn.Identity()
        self.detection_head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        features = self.image_encoder(x)
        rois = self.detection_decoder(features)
        outputs = self.detection_head(rois)
        return outputs
