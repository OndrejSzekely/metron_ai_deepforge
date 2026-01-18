# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2026 Ondrej Szekely (ondra.szekely@gmail.com).

from torch import nn

import metron_shared.param_validators as param_val
from metron_shared.utils import is_debug_enabled


class LayersInspectorMixin:
    """Mixin class for inspection of all <nn.Module> class instance attributes to get specific layers."""

    def search_for_layer_by_name(self, layer_name: str) -> list[nn.Module]:
        """Searches for a layer by its name in class instance <nn.Module> attributes."""
        if is_debug_enabled():
            assert isinstance(layer_name, str)

        attributes = vars(self)

        found_layers: list[nn.Module] = []
        for _, attr_value in attributes.items():
            if param_val.check_type(attr_value, nn.Module):
                for name, layer in attr_value.named_modules():
                    if name == layer_name:
                        found_layers.append(layer)

        return found_layers
