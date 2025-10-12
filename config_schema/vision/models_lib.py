# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Defines Hydra's Structured Config schema for vision model configuration files."""

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class VisionModelBaseConfig:
    """Base configuration for vision models."""

    _target_: str = MISSING


@dataclass
class DiffusionDetConfig(VisionModelBaseConfig):
    """Configuration for DiffusionDet model."""

    _target_: str = "forge.vision.models.diffusiondet.DiffusionDet"


def register_lib() -> None:
    """Register vision model configurations."""

    cs = ConfigStore.instance()
    cs.store(
        group="deepforge/vision/models_lib",
        name="diffusiondet",
        node=DiffusionDetConfig,
        package="deepforge.vision.models_lib",
    )
