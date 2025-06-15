# This file is part of the Metron AI ArDaGen (https://github.com/OndrejSzekely/metron_ai_ardagen).
# Copyright (c) 2025 Ondrej Szekely.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3. This program
# is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. You should have received a copy of the GNU General Public
# License along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Miscelanneous logging functions."""

import logging


def log_kfold_split_info(train_images_num: int, val_images_num: int) -> None:
    logging.info("*" * 15 + " SPLIT INFO " + "*" * 15)
    logging.info(f"Training images: {train_images_num}")
    logging.info(f"Val images: {val_images_num}")
    logging.info("*" * 42)
