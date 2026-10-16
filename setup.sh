#!/usr/bin/env bash
# One-time setup: virtual env, dependencies and model weights, all inside this folder.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-$(command -v python3.12 || command -v python3)}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
mkdir -p img/in img/out

echo "Downloading model weights into ./models (about 1.2 GB, first time only)..."
HF_HUB_OFFLINE=0 .venv/bin/python -W ignore - <<'PY'
import env  # keeps downloads inside ./models
import logging
logging.basicConfig(level=logging.INFO, format="  %(message)s")
from segment import DogSegmenter
import edgemaps
import features
import depth
DogSegmenter(device="cpu")
for name in edgemaps.ML_MODELS:
    edgemaps.load_model(name, "cpu")
features.FeatureFinder("cpu")
depth.load_model("cpu")
PY

echo
echo "Done. Put photos in img/in/ and run:"
echo "  .venv/bin/python dog_lineart.py --style tattoo --preview"
