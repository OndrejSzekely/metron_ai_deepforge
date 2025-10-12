#!/bin/sh

# Install pre-commit hooks
cd /metron_ai_deepforge_repo && uv pre-commit install

# Initialize Metron Shared submodule
cd /metron_ai_deepforge_repo && git submodule update --init --recursive

# Add Metron AI DeepForge folder into Python PATH
echo "export PYTHONPATH=/metron_ai_deepforge_repo:$PYTHONPATH" >> "$BASHRC"