#!/bin/sh

# Install pre-commit hooks
cd /metron_ai_deepforge_repo && pre-commit install

# Install Metron Shared submodule
cd /metron_ai_deepforge_repo && pip install -e metron_shared

# Add Metron AI DeepForge folder into Python PATH
echo "export PYTHONPATH=/metron_ai_deepforge_repo:$PYTHONPATH" >> "$BASHRC"