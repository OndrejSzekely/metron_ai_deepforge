#!/bin/sh

# Install pre-commit hooks
cd /metron_ai_deepforge_repo && pre-commit install

# Install Metron Shared submodule
cd /metron_ai_deepforge_repo && pip install -e metron_shared