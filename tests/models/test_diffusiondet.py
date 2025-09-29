# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Tests for DiffusionDet model."""

import pytest

import metron_shared.param_validators as param_val
from forge.models.diffusiondet import DiffusionDet


@pytest.mark.unit
def test_diffusiondet_initialization() -> None:
    """Test the initialization of the DiffusionDet model."""

    # GIVEN DiffusionDet model parameters

    # WHEN initializing the DiffusionDet mode
    model = DiffusionDet()

    # THEN the model shall be initialized correctly
    param_val.check_type(model, DiffusionDet)
