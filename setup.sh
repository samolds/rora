#!/usr/bin/env bash
# One-time setup: virtual env, dependencies and model weights, all inside this folder.
# ./setup.sh --refine also downloads the image-editing model for the optional --refine pass.
set -euo pipefail
cd "$(dirname "$0")"

REFINE=0
for arg in "$@"; do
  case "$arg" in
    --refine) REFINE=1 ;;
    *) echo "usage: ./setup.sh [--refine]" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-$(command -v python3.12 || command -v python3)}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

[ -d env ] || "$PY" -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install -r requirements.txt
mkdir -p img/in img/out

echo "Downloading model weights into ./models (about 1.2 GB, first time only)..."
HF_HUB_OFFLINE=0 env/bin/python -W ignore - <<'PY'
from rora import env  # keeps downloads inside ./models
import logging
logging.basicConfig(level=logging.INFO, format="  %(message)s")
from rora.segment import Segmenter
from rora import depth, edgemaps, features
Segmenter(device="cpu")
for name in edgemaps.ML_MODELS:
    edgemaps.load_model(name, "cpu")
features.FeatureFinder("cpu")
depth.load_model("cpu")
PY

if [ "$REFINE" = 1 ]; then
  echo "Downloading FLUX.2 [klein] 4B for --refine into ./models (about 16 GB, first time only)..."
  HF_HUB_OFFLINE=0 env/bin/python -W ignore -c 'from rora import refine; refine.download()'
fi

echo
echo "Done. Put photos in img/in/ and run:"
echo "  env/bin/python dog_lineart.py --style tattoo --preview"
if [ "$REFINE" = 0 ]; then
  echo
  echo "Optional: run ./setup.sh --refine to enable the --refine pass (downloads about 16 GB)."
fi
