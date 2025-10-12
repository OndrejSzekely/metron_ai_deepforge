# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Defines Hydra's Structured Config schema for vision encoders configuration files."""

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class VisionEncoderBaseConfig:
    """Base configuration for vision encoders."""

    _target_: str = MISSING


@dataclass
class ResNet50Config(VisionEncoderBaseConfig):
    """Configuration for ResNet-50 encoder."""

    _target_: str = "forge.vision.encoders.resnet50.ResNet50"


def register_lib() -> None:
    """Register vision encoder configurations."""

    cs = ConfigStore.instance()
    cs.store(
        group="deepforge/vision/encoders_lib",
        name="resnet50",
        node=ResNet50Config,
        package="deepforge.vision.encoders_lib",
    )
