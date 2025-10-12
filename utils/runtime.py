# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Runtime utilities for DeepForge."""

import os
import pkgutil
from functools import wraps
from typing import Any, Callable


def register_all_config_schema_libs(main_function: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to register all configuration schema libraries in <config_schema> package before Hydra's OmegaConf is composed.

    Args:
        main_function (Callable[..., Any]): Main function which performs program orchestration.

    Returns (Callable[..., Any]): Decorated <main_function>.

    Raises:
        AttributeError: If any module in <config_schema> package does not have <register_lib> function.
    """

    @wraps(main_function)
    def decorated_function(*args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """
        Decorating function which calls all the registration functions from <config_schema> package.

        Args:
            args (list[Any]): Positional arguments for <main_function>.
            kwargs (dict[str, Any]): Keyword arguments for <main_function>.

        Returns (Any): Return value of <main_function>.

        Raises:
            AttributeError: If any module in <config_schema> package does not have <register_lib> function.
        """

        for module_info in pkgutil.walk_packages(
            path=[f"./{os.environ['DEEPFORGE_STRUCTURED_CONFIG_SCHEMAS_PACKAGE']}"],
            prefix=f"{os.environ['DEEPFORGE_STRUCTURED_CONFIG_SCHEMAS_PACKAGE']}.",
        ):
            if not module_info.ispkg:
                lib = __import__(module_info.name, fromlist=["register_lib"])
                if not hasattr(lib, "register_lib"):
                    raise AttributeError(
                        f"Config schema module <{module_info.name}> does not have <register_lib> function.",
                    )
                lib.register_lib()

        return main_function(*args, **kwargs)

    return decorated_function
