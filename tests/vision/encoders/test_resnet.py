# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests implementation for `resnet` module."""

from typing import Optional

import pytest

from config_schema.vision.encoders_lib import ResNetType
from forge.vision.encoders.resnet import ResNet
from metron_shared import param_validators as param_val

test_resnet_types = [item for item in ResNetType]
test_resnet_init_weights = [None]


@pytest.mark.unit
@pytest.mark.parametrize(
    "resnet_version",
    test_resnet_types,
)
@pytest.mark.parametrize("weights_type", test_resnet_init_weights)
def test_initialize_resnet_class(resnet_version: ResNetType, weights_type: Optional[str]):
    """Testing <ResNet> class constructor"""
    # GIVEN: <ResNet> class constructor parameters

    # WHEN: Construcor is called
    resnet = ResNet(resnet_version, weights_type)

    # THEN: Instance is created
    assert param_val.check_type(resnet, ResNet)


@pytest.mark.unit
def test_initialize_resnet_class_with_non_existing_weights():
    """Testing <ResNet> class constructor with non-existing weights"""
    # GIVEN: <ResNet> class constructor parameters and non-existing init weights name
    resnet_version = ResNetType.ResNet18
    weights_name = "non-existing"

    # WHEN: Constructor is called
    # THEN: Error is raised
    with pytest.raises(ValueError):
        ResNet(resnet_version, weights_name)
