#!/usr/bin/env bash
# ==============================================================================
# Body2Fit Web Demo Launcher
# Starts the high-performance standalone web demo with live PyTorch inference,
# Three.js 3D mesh rendering, and clinical cardiometabolic risk interpretation.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1:-8080}"
HOST="${2:-0.0.0.0}"
CKPT="checkpoints/best_640x480_v4_resnet.pt"

echo "================================================================================"
echo "🚀 Starting Body2Fit Anthropometry & Health Screening Demo"
echo "================================================================================"

# Check checkpoint
if [ ! -f "$CKPT" ]; then
    echo "❌ Error: Model checkpoint not found at $CKPT"
    exit 1
fi

# Locate Python environment
if [ -d ".venv" ] && [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
    echo "✓ Using project virtualenv: $PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
    echo "✓ Using system python3"
else
    echo "❌ Error: Python 3 not found."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "✓ Checkpoint: $CKPT"
echo "✓ Serving on: http://localhost:${PORT}"
echo "✓ API Health: http://localhost:${PORT}/api/health"
echo "✓ API Predict: http://localhost:${PORT}/api/predict"
echo "================================================================================"
echo "Press Ctrl+C to stop the server."
echo ""

exec "$PYTHON" web/server.py --port "$PORT" --host "$HOST" --ckpt "$CKPT"
