#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
SAM2_CHECKPOINT="${ROOT_DIR}/models/segmentation/sam2.1_hiera_large.pt"
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Project .venv is required at ${VENV_PYTHON}" >&2
  exit 1
fi

if [[ "${VIRTUAL_ENV:-}" != "${ROOT_DIR}/.venv" ]]; then
  echo "Activate the project environment first: source .venv/bin/activate" >&2
  exit 1
fi

cd "${ROOT_DIR}"

if command -v uv >/dev/null 2>&1; then
  SAM2_BUILD_CUDA=0 uv pip install --python "${VENV_PYTHON}" -r requirements-inference.txt
else
  SAM2_BUILD_CUDA=0 "${VENV_PYTHON}" -m pip install -r requirements-inference.txt
fi

mkdir -p "$(dirname "${SAM2_CHECKPOINT}")"
if [[ ! -f "${SAM2_CHECKPOINT}" ]]; then
  curl -L --fail --retry 3 -o "${SAM2_CHECKPOINT}" "${SAM2_URL}"
fi

"${VENV_PYTHON}" - <<'PY'
from pathlib import Path

required = [
    "models/segmentation/yolo11m.pt",
    "models/segmentation/sam2.1_hiera_large.pt",
    "configs/segmentation/sam2.1/sam2.1_hiera_l.yaml",
    "models/nlf/nlf_l_multi.torchscript",
    "models/smplx/SMPLX_NEUTRAL.npz",
]
missing = [path for path in required if not Path(path).exists()]
if missing:
    raise SystemExit("Missing required inference assets: " + ", ".join(missing))

for module_name in ["ultralytics", "sam2", "smplx", "trimesh", "pyrender", "torchvision"]:
    __import__(module_name)
print("Strict inference dependencies are ready.")
PY
