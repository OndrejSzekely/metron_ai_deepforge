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

SHAKESPEARE_INPUT_TEXT: str = "/mnt/datastore/datasets/sample/tinyshakespeare/input.txt"

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    """Main function to run the script."""

    # Read Shakespeare text from a file into <text>
    with open(SHAKESPEARE_INPUT_TEXT, "r", encoding="utf-8") as f:
        text = f.read()

    # Get vocabulary size
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    logger.info(f"Vocabulary size: {vocab_size}")


if __name__ == "__main__":
    main()
