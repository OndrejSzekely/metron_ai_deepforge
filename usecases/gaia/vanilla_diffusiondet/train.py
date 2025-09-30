# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""Training script for Vanilla DiffusionDet model of Gaia project."""

import hydra
from omegaconf import DictConfig

from metron_shared.config.config import set_hydra_config


@hydra.main(version_base="1.3", config_path=".", config_name="config")
@set_hydra_config
def main(cfg: DictConfig) -> None: ...


if __name__ == "__main__":
    main()
