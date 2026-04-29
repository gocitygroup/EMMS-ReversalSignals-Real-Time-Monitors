#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  EMMS + Reversal Signals — Environment Setup"
echo "============================================================"
echo

# -- check python --
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.10+ first."
    exit 1
fi
echo "[OK] Found $(python3 --version)"

# -- create venv --
if [ ! -f ".venv/bin/python" ]; then
    echo
    echo "[INFO] Creating virtual environment .venv ..."
    python3 -m venv .venv
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment .venv already exists."
fi

# -- activate & upgrade pip --
source .venv/bin/activate
pip install --upgrade pip --quiet

# -- install core dependencies --
echo
echo "[INFO] Installing core dependencies ..."
pip install -r requirements.txt || echo "[WARN] MetaTrader5 may not be available on this platform (Windows-only)."

# -- offer socketio --
echo
read -r -p "Install optional Socket.IO packages? (y/n): " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    echo "[INFO] Installing Socket.IO packages ..."
    pip install -r requirements-socketio.txt
fi

echo
echo "============================================================"
echo "  Setup complete."
echo
echo "  Activate the environment before running scripts:"
echo "    source .venv/bin/activate"
echo
echo "  Run the monitors:"
echo "    python EstimatedManipulationMovementSignal.py --realtime"
echo "    python ReversalSignals.py --realtime"
echo "============================================================"
