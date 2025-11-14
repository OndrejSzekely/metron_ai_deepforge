# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Defines Hydra's Structured Config schema for vision encoders configuration files."""

from dataclasses import dataclass
from enum import StrEnum

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class ResNetType(StrEnum):
    ResNet18 = "resnet18"
    ResNet34 = "resnet34"
    ResNet50 = "resnet50"
    ResNet101 = "resnet101"
    ResNet152 = "resnet152"


@dataclass
class VisionEncoderBaseConfig:
    """Base configuration for vision encoders."""

    _target_: str = MISSING


@dataclass
class ResNetConfig(VisionEncoderBaseConfig):
    """Configuration for ResNet encoder."""

    resnet_version: ResNetType = MISSING
    init_weights_type: str | None = None
    _target_: str = "forge.vision.encoders.resnet.ResNet"


def register_lib() -> None:
    """Register vision encoder configurations."""

    cs = ConfigStore.instance()
    cs.store(
        group="deepforge/vision/encoders_lib",
        name="resnet",
        node=ResNetConfig,
        package="deepforge.vision.encoders_lib",
    )
