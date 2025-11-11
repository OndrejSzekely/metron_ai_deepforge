# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""ResNet Encoder for Vision Tasks."""

import re
from dataclasses import dataclass
from importlib import import_module
from typing import Optional

from torch import Tensor, nn

from config_schema.vision.encoders_lib import ResNetType
from forge.common.data_structures import HWC
from metron_shared import param_validators as param_val
from metron_shared.utils import is_debug_enabled


@dataclass
class ResNetAttachmentPoint:
    attachment_layer: nn.Module
    layer_name: str
    output_stride: int
    output_resolution: HWC


class ResNet(nn.Module):
    """ResNet Encoder for Vision Tasks."""

    _layer_name_regex = r"layer\d+$"  # matches only `layer<number>`

    def __init__(self, resnet_version: ResNetType, init_weights_type: Optional[str], input_resolution: Optional[HWC] = None) -> None:
        if is_debug_enabled():
            assert param_val.check_type(resnet_version, ResNetType)
            assert param_val.check_type(init_weights_type, str | None)
            assert param_val.check_type(input_resolution, Optional[HWC])
        super().__init__()
        self.input_resolution = input_resolution
        self._resnet_model = self._instantiate_resnet_model(resnet_version, init_weights_type)
        self.attachment_points = []

    def _instantiate_resnet_model(self, resnet_version: ResNetType, init_weights_type: Optional[str]) -> nn.Module:
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
                layers.append(ResNetAttachmentPoint(attachment_layer=layer, layer_name=layer_name, output_resolution=HWC(), output_stride=0))
        # TODO: Enrich resolution and output stride.
        return layers

    def set_attachment_point(
        self, layer_name: Optional[str] = None, output_resolution: Optional[HWC] = None, output_stride: Optional[int] = None
    ) -> None: ...

    def forward(self, x: Tensor):
        if is_debug_enabled():
            if self.input_resolution:
                assert HWC(H=x.size(2), W=x.size(3), C=x.size(1)) == self.input_resolution
        output = []
        for module_name, module in self._resnet_model.named_children():
            x = module(x)
            if module_name in self.attachment_points:
                output.append(x)
