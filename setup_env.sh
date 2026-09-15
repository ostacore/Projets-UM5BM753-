#!/usr/bin/env bash
# Creates (if needed) and syncs the project's virtual environment from requirements.txt.
# Usage: ./setup_env.sh
set -e
cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating .venv ..."
    uv venv
fi

echo "Installing packages from requirements.txt ..."
uv pip install -r requirements.txt --python .venv/bin/python

echo "Done. Activate with: source .venv/bin/activate"
