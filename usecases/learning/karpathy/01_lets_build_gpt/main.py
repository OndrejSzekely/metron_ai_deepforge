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
from utils import BigramLanguageModel, BigramLanguageModelBase, Dataset, Split, estimate_loss

SHAKESPEARE_INPUT_TEXT: str = "datasets/sample/tinyshakespeare/input.txt"
TRAIN_DATA_FRACTION: float = 0.9
BLOCK_SIZE: int = 8
BATCH_SIZE: int = 4
EVAL_ITERS: int = 200
EMBEDDING_DIM: int = 32

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    itos = {i: ch for i, ch in enumerate(chars)}

    # Define encoding/decoding functions
    encode = lambda s: [stoi[c] for c in s]  # noqa: E731
    decode = lambda l: "".join([itos[i] for i in l])  # noqa: E731, E741

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

    # Create dataset generator
    dataset = Dataset(data, split_idx, BLOCK_SIZE, BATCH_SIZE, device)
    xb, yb = dataset.get_batch(Split.TRAIN)
    logger.info("Sample input:")
    logger.info(f"shape {xb.shape}")
    logger.info(f"values {xb}")
    logger.info("Sample target:")
    logger.info(f"shape {yb.shape}")
    logger.info(f"values {yb}")

    # First model: BigramLanguageModelBase
    logger.info("Simple BigramLanguageModelBase...")
    torch.manual_seed(1337)
    bigram_model = BigramLanguageModelBase(vocab_size, EMBEDDING_DIM)
    bigram_model.to(device)
    logits, loss = bigram_model(xb, yb)
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Loss: {loss.item()}")

    # Generate new tokens
    seed_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_text = decode(bigram_model.generate(seed_token, max_new_tokens=100)[0].tolist())
    logger.info("Generated text of untrained model:")
    logger.info(gen_text)

    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)
    for step in range(10000):
        xb, yb = dataset.get_batch(Split.TRAIN)
        logits, loss = bigram_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_ITERS == 0:
            losses = estimate_loss(EVAL_ITERS, bigram_model, dataset)
            logger.info(f"Step {step}, train loss: {losses[Split.TRAIN]:.4f}, val loss: {losses[Split.VALIDATION]:.4f}")

    # Generate new tokens from trained model
    seed_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_text = decode(bigram_model.generate(seed_token, max_new_tokens=400)[0].tolist())
    logger.info("Generated text of trained model:")
    logger.info(gen_text)

    # Second model: BigramLanguageModel
    logger.info("Simple BigramLanguageModel...")
    torch.manual_seed(1337)
    bigram_model = BigramLanguageModel(vocab_size, EMBEDDING_DIM, BLOCK_SIZE)

    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)
    for step in range(10000):
        xb, yb = dataset.get_batch(Split.TRAIN)
        logits, loss = bigram_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_ITERS == 0:
            losses = estimate_loss(EVAL_ITERS, bigram_model, dataset)
            logger.info(f"Step {step}, train loss: {losses[Split.TRAIN]:.4f}, val loss: {losses[Split.VALIDATION]:.4f}")


if __name__ == "__main__":
    main()
