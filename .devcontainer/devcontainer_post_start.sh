#!/bin/bash

# Install dependencies
sudo uv sync --all-groups  --inexact 

# Install pre-commit hooks
pre-commit install

# Load all secret enviromental variables
SECRET_ENV_DIR="./.devcontainer/secrets"
BASHRC="$HOME/.bashrc"

if [ ! -d "$SECRET_ENV_DIR" ]; then
  echo "Secrets directory not found: $SECRET_ENV_DIR"
fi
echo "Appending secret env variables into $BASHRC"

for env_file in "$SECRET_ENV_DIR"/*.env; do
  [ -e "$env_file" ] || continue  # skip if no matching files
  echo "Processing secret env file $env_file..."

  # Make sure a trailing newline is in available, otherwise `read` tool will fail 
  sed -i -e '$a\' $env_file

  # Read each non-empty, non-comment line
  while IFS='=' read -r key value; do
      # Skip blank lines and comments
      [[ "$key" =~ ^\s*# ]] || [[ -z "$key" ]] && continue

      # Remove surrounding quotes from value (if any)
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"

      # Escape any double quotes in value
      value="${value//\"/\\\"}"

      # Append to .bashrc
      echo "export $key=\"$value\"" >> "$BASHRC"
  done < "$env_file"
done