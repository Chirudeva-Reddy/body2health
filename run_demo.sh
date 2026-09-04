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
CKPT_URL="https://github.com/Chirudeva-Reddy/body2health/releases/download/v1.0.0-weights/best_640x480_v4_resnet.pt"
CKPT_SHA256="7fc383762e98367101243806daaacf32105e9cecbab5ed9f9be780dc13563631"

# The checkpoint is 136 MB, so it ships as a release asset rather than in git.
if [ ! -f "$CKPT" ]; then
    echo "Model checkpoint not found. Downloading it once (136 MB)..."
    mkdir -p "$(dirname "$CKPT")"
    if ! curl -fL --progress-bar "$CKPT_URL" -o "$CKPT.part"; then
        rm -f "$CKPT.part"
        echo "Download failed. Fetch it manually from:"
        echo "  $CKPT_URL"
        echo "and save it to $CKPT"
        exit 1
    fi
    mv "$CKPT.part" "$CKPT"
    echo "Downloaded to $CKPT"
fi

# Verify the checkpoint rather than trusting whatever is on disk.
if command -v shasum >/dev/null 2>&1; then
    GOT="$(shasum -a 256 "$CKPT" | awk '{print $1}')"
    if [ "$GOT" != "$CKPT_SHA256" ]; then
        echo "Checkpoint at $CKPT failed its checksum."
        echo "  expected $CKPT_SHA256"
        echo "  got      $GOT"
        echo "Delete the file and re-run to download a clean copy."
        exit 1
    fi
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
