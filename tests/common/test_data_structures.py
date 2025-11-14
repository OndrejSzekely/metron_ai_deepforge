# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests implementation for `data_structures` module."""

from dataclasses import dataclass

import pytest

from forge.common.data_structures import NullSizeValue, _DimensionSizeDescriptor
from metron_shared import param_validators as param_val


@dataclass
class _TestDimensionSizeDescriptorClass:
    """Dummy test dataclass using <_DimensionSizeDescriptor> descriptor."""

    dimsize: _DimensionSizeDescriptor = _DimensionSizeDescriptor()


@pytest.mark.unit
def test_dimensionsizedesecriptor_default_value():
    """Test default value `NullSizeValue` is returned for <_DimensionSizeDescriptor> dataclass attribute."""
    # GIVEN: <_TestDimensionSizeDescriptorClass> instance with no initialization values
    dummy_dimension_size_class = _TestDimensionSizeDescriptorClass()

    # WHEN: <dimsize> attribute is accessed
    # THEN: its default value is `NullSizeValue`
    assert param_val.check_type(dummy_dimension_size_class.dimsize, NullSizeValue)
    assert dummy_dimension_size_class.dimsize == NullSizeValue()


@pytest.mark.unit
def test_dimensionsizedesecriptor_non_default_value():
    """Test non-default value is returned for <_DimensionSizeDescriptor> dataclass attribute."""
    # GIVEN: <_TestDimensionSizeDescriptorClass> instance with init value <value>
    value = 4
    dummy_dimension_size_class = _TestDimensionSizeDescriptorClass(value)

    # WHEN: <dimsize> attribute is accessed
    # THEN: its value is equal to <value>
    assert dummy_dimension_size_class.dimsize == value
