# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""SearchPathPlugin implementation for adding DeepForge's specific search paths."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class DeepForgeSearchPathPlugin(SearchPathPlugin):
    """SearchPathPlugin implementation for adding DeepForge's specific search paths."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Add DeepForge-specific search paths to Hydra's configuration search path.

        Args:
            search_path (ConfigSearchPath): The current configuration search path.
        """
        search_path.append(provider="deepforge-searchpath-plugin", path="pkg://config_schema")
        search_path.append(provider="deepforge-searchpath-plugin", path="pkg://config")
