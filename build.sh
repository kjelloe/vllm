#!/bin/bash
# Build vllm: compiles the Rust frontend then installs the Python package.
# Usage: ./build.sh [--debug]
#
# Pass --debug to build the Rust binary in debug mode (faster compile).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Build Rust frontend first; forward any flags (e.g. --debug).
"$REPO_ROOT/build_rust.sh" "$@"

# Ensure uv is available.
if ! command -v uv &>/dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
    echo "Note: uv was just installed. If subsequent commands fail, open a new shell or run: source \"\$HOME/.cargo/env\""
fi

# Create/update the Python 3.12 virtual environment.
uv venv --python 3.12

# Install vllm in editable mode using the precompiled Rust binary.
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
