#!/bin/bash

# Install dependencies
sudo uv sync --all-groups  --inexact 

# Install pre-commit hooks
pre-commit install