# This file is part of the Metron AI ArDaGen (https://github.com/OndrejSzekely/metron_ai_ardagen).
# Copyright (c) 2025 Ondrej Szekely.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3. This program
# is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. You should have received a copy of the GNU General Public
# License along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Lecture by Andrej Karpathy on building a GPT-like model from scratch."""

import logging
import os

import torch

SHAKESPEARE_INPUT_TEXT: str = "datasets/sample/tinyshakespeare/input.txt"
TRAIN_DATA_FRACTION: float = 0.9

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    """Main function to run the script."""

    # Read Shakespeare text from a file into <text>
    with open(os.path.join(os.environ["METRON_AI_CATALOGUE_PATH"], SHAKESPEARE_INPUT_TEXT), "r", encoding="utf-8") as f:
        text = f.read()

    # Get vocabulary size
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create characters <-> integers mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    # itos = {i: ch for i, ch in enumerate(chars)}

    # Define encoding/decoding functions
    encode = lambda s: [stoi[c] for c in s]  # noqa: E731
    # decode = lambda l: "".join([itos[i] for i in l])

    # Encode the text into tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    logger.info(f"Data tensor shape: {data.shape}")
    logger.info(f"Data tensor type: {data.dtype}")

    # Data split
    split_idx = int(len(data) * TRAIN_DATA_FRACTION)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    logger.info(f"Train data size: {len(train_data)}")
    logger.info(f"Validation data size: {len(val_data)}")


if __name__ == "__main__":
    main()
