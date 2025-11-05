# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Common reusable data structures."""

from dataclasses import dataclass
from typing import Any


class _DimensionSizeDescriptor:
    """Generic positive dimension size descriptor. Default value is `0`, indicating the attribute is not relevant.

    This allows to use the descriptor in generic dataclasses e.g. <HWC> where `C` channels is not relevant everytime and it prevents necessity to
    create a special case class.
    """

    _attribute_name = "_dimension"

    def __get__(self, instance: Any, owner: Any) -> int:
        if instance is None:
            raise AttributeError("Class attribute access is not allowed.")

        return getattr(instance, _DimensionSizeDescriptor._attribute_name, 0)

    def __set__(self, instance: Any, value: int) -> None:
        if value <= 0:
            raise ValueError(f"Image dimension can not be set to zero or negative value. You set `{value}`")

        setattr(instance, _DimensionSizeDescriptor._attribute_name, value)


@dataclass
class HWC:
    H: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
    W: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
    C: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
