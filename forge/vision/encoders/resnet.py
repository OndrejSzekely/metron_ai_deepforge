# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""ResNet Encoder for Vision Tasks."""

from importlib import import_module
from typing import Optional

from torch import nn

from config_schema.vision.encoders_lib import ResNetType
from metron_shared import param_validators as param_val
from metron_shared.utils import is_debug_enabled


class ResNet(nn.Module):
    """ResNet Encoder for Vision Tasks."""

    def _instantiate_resnet_model(self, resnet_version: ResNetType, init_weights_type: Optional[str]) -> None:
        _module = "torchvision.models"

        if is_debug_enabled():
            assert param_val.check_type(resnet_version, ResNetType)
            assert param_val.check_type(init_weights_type, str | None)

        models_module = import_module(_module)
        if not hasattr(models_module, resnet_version.value):
            raise ValueError(f"Module `{_module}` has not given ResNet class <{resnet_version.value}>.")
        resnet_class = getattr(models_module, resnet_version.value)

        return resnet_class()

    def __init__(self, resnet_version: ResNetType, init_weights_type: Optional[str]) -> None:
        if is_debug_enabled():
            assert param_val.check_type(resnet_version, ResNetType)
            assert param_val.check_type(init_weights_type, str | None)
        super().__init__()

        self.resnet_model = self._instantiate_resnet_model(resnet_version, init_weights_type)
