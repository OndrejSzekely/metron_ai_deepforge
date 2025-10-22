# This file is part of the Metron AI DeepForge (https://github.com/OndrejSzekely/metron_ai_deepforge).
# Copyright (c) 2025 Ondrej Szekely (ondra.szekely@gmail.com).

"""I/O utlilities"""

import os

CHEKPOINTS_DIR_NAME = "checkpoints"
VISU_DIR_NAME = "visu"


def create_model_run_output_dir(base_output_dir: str) -> str:
    """Create output directory for a specific model run.

    Args:
        base_output_dir (str): Base output directory.

    Returns:
        str: Path to the created output directory.
    """

    os.makedirs(base_output_dir, exist_ok=True) if not os.path.exists(base_output_dir) else None
    runs_num = len(os.listdir(base_output_dir))
    run_id = runs_num - 1 if runs_num > 0 else 0
    output_dir = os.path.join(base_output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, CHEKPOINTS_DIR_NAME))
    os.makedirs(os.path.join(output_dir, VISU_DIR_NAME))
    return output_dir
