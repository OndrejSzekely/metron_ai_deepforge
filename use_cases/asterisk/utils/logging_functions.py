"""Miscelanneous logging functions.
"""

import logging


def log_kfold_split_info(train_images_num: int, val_images_num: int) -> None:
    logging.info("*" * 15 + " SPLIT INFO " + "*" * 15)
    logging.info(f"Training images: {train_images_num}")
    logging.info(f"Val images: {val_images_num}")
    logging.info("*" * 42)
