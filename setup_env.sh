#!/usr/bin/env bash
set -euo pipefail

# setup_env.sh
# Create a virtual environment and install the project's core Python dependencies.
# Usage:
#   bash setup_env.sh           # uses 'python3' from PATH
#   bash setup_env.sh /usr/bin/python3.10  # use a specific python executable

VENV_DIR=".venv"
PYTHON="${1:-python3}"
PACKAGES=(numpy matplotlib pandas huggingface_hub)

echo "[setup] Using Python: $PYTHON"

# Ensure the python executable exists
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python executable '$PYTHON' not found." >&2
  exit 2
fi

# Create venv if missing
if [ -d "$VENV_DIR" ]; then
  echo "[setup] Virtual environment '$VENV_DIR' already exists. Re-using it."
else
  echo "[setup] Creating virtual environment in '$VENV_DIR'..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# Use venv's python/pip to upgrade pip and install packages
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

echo "[setup] Upgrading pip in venv..."
"$VENV_PY" -m pip install --upgrade pip

echo "[setup] Installing packages: ${PACKAGES[*]}"
"$VENV_PIP" install "${PACKAGES[@]}"

cat <<EOF

[setup] Done.
To activate the virtual environment run:
  source $VENV_DIR/bin/activate
Then run the AE script:
  python main_ae.py

If you prefer system packages on Debian/Ubuntu, run:
  sudo apt update && sudo apt install python3-numpy python3-matplotlib python3-pandas python3-venv

EOF
