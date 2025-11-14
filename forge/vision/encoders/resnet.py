# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""ResNet Encoder for Vision Tasks."""

import re
from dataclasses import dataclass
from importlib import import_module

from torch import Tensor, nn

from config_schema.vision.encoders_lib import ResNetType
from forge.common.data_structures import CHW
from metron_shared import param_validators as param_val
from metron_shared.utils import is_debug_enabled


@dataclass
class ResNetAttachmentPoint:
    """Attachment point info"""

    attachment_layer: Tensor | nn.Module
    layer_name: str
    output_stride: int
    output_resolution: CHW


class ResNet(nn.Module):
    """ResNet Encoder for Vision Tasks.

    Attributes:
        input_resolution (CHW | None): Input resolution. If not set, no resolution dependent info is returned from  <get_attachment_layers()>
        _resnet_model (nn.Module): Torchvision model.
        _attachment_points (list[Tensor | nn.Module]): List of attachment points for which outputs are returned in <forward()> method
    """

    _layer_name_regex = r"layer\d+$"  # matches only `layer<number>`

    def __init__(self, resnet_version: ResNetType, *, init_weights_type: str | None = None, input_resolution: CHW | None = None) -> None:
        """N/A

        Args:
            resnet_version (ResNetType): Torchvision ResNet type given by https://docs.pytorch.org/vision/main/models/resnet.html
            init_weights_type (str | None): Init weights type to be loaded. Defaults to None.
            input_resolution (CHW | None): Input resolution. Defaults to None.
        """
        if is_debug_enabled():
            assert param_val.check_type(resnet_version, ResNetType)
            assert param_val.check_type(init_weights_type, str | None)
            assert param_val.check_type(input_resolution, CHW | None)
        super().__init__()
        self.input_resolution = input_resolution
        self._resnet_model = self._instantiate_resnet_model(resnet_version, init_weights_type)
        self._attachment_points = []

    def _instantiate_resnet_model(self, resnet_version: ResNetType, init_weights_type: str | None) -> nn.Module:
        """Instantiates Torchvision model.

        Args:
            resnet_version (ResNetType): Torchvision ResNet type given by https://docs.pytorch.org/vision/main/models/resnet.html
            init_weights_type (str | None): Init weights type to be loaded. Defaults to None.

        Raises:
            ValueError: Raised when given non-existing ResNet version.
            ValueError: Raised wehen given non-existing ResNet init weights name.

        Returns:
            nn.Module: Initialized Resnet
        """
        _module = "torchvision.models"

        if is_debug_enabled():
            assert param_val.check_type(resnet_version, ResNetType)
            assert param_val.check_type(init_weights_type, str | None)

        models_module = import_module(_module)
        if not hasattr(models_module, resnet_version.value):
            raise ValueError(f"Module `{_module}` has not given ResNet class <{resnet_version.value}>.")
        resnet_class = getattr(models_module, resnet_version.value)

        try:
            return resnet_class(init_weights_type)
        except KeyError as e:
            raise ValueError(f"Given non-existing ResNet weights name `{init_weights_type}`.") from e

    def get_attachment_layers(self) -> list[ResNetAttachmentPoint]:
        layers = []
        for layer_name, layer in self._resnet_model.named_modules():
            if re.match(ResNet._layer_name_regex, layer_name):
                layers.append(ResNetAttachmentPoint(attachment_layer=layer, layer_name=layer_name, output_resolution=CHW(), output_stride=0))
        # TODO: Enrich resolution and output stride.
        return layers

    def set_attachment_point(self, layer_name: str | None = None, output_resolution: CHW | None = None, output_stride: int | None = None) -> None: ...

    def forward(self, x: Tensor):
        """N/A

        Args:
            x (Tensor): Input image in forme (B,C,H,W)

        Returns:
            (list[Tensor]): List of attachment points output tensors (B,C,H,W)
        """
        if is_debug_enabled():
            if self.input_resolution:
                assert CHW(C=x.size(1), H=x.size(2), W=x.size(3)) == self.input_resolution
        output = []
        for module_name, module in self._resnet_model.named_children():
            x = module(x)
            if module_name in self._attachment_points:
                output.append(x)
