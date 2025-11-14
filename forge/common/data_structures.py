# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Common reusable data structures."""

from dataclasses import dataclass, field
from typing import Any

from metron_shared import param_validators as param_val


class NullSizeValue(int):
    """It's not comparable with any value, except <__eq__>, <__ne__> operands. Otherwise, it raises `ValueError"""

    def __repr__(self):
        return "NullSizeValue"

    def __eq__(self, value):
        if param_val.check_type(value, NullSizeValue):
            return True
        return False

    def __ne__(self, value):
        if param_val.check_type(value, NullSizeValue):
            return False
        return True

    def __le__(self, value):
        raise TypeError("`NullSizeValue` can not be used in any operand except <__eq__>, <__ne__>")

    def __lt__(self, value):
        raise TypeError("`NullSizeValue` can not be used in any operand except <__eq__>, <__ne__>")

    def __ge__(self, value):
        raise TypeError("`NullSizeValue` can not be used in any operand except <__eq__>, <__ne__>")

    def __gt__(self, value):
        raise TypeError("`NullSizeValue` can not be used in any operand except <__eq__>, <__ne__>")


class _DimensionSizeDescriptor:
    """Generic positive dimension size descriptor. Default returned value is `NullSizeValue()` for dataclass.
    See, how Descriptors are handled in dataclasses: https://docs.python.org/3/library/dataclasses.html
    """

    _name = "_dim_size"

    def __get__(self, instance: Any, owner: Any) -> int | NullSizeValue:
        """See https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields, bullet-point 3."""
        if instance is None:
            return field(default_factory=NullSizeValue)
        return getattr(instance, _DimensionSizeDescriptor._name)

    def __set__(self, instance: Any, value: int) -> None:
        if value <= 0 and not param_val.check_type(value, NullSizeValue):
            raise ValueError(f"Dimension size can not be set to zero or negative value. You set `{value}`")

        setattr(instance, _DimensionSizeDescriptor._name, value)


@dataclass
class CHW:
    """Default value is `NullSizeValue`, indicating the attribute is not set."""

    C: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
    H: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
    W: _DimensionSizeDescriptor = _DimensionSizeDescriptor()


@dataclass
class BCHW:
    """Default value is `NullSizeValue`, indicating the attribute is not set."""

    B: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
    C: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
    H: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
    W: _DimensionSizeDescriptor = _DimensionSizeDescriptor()
