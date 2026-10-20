#!/usr/bin/env bash
# One-time setup: virtual env, dependencies and model weights, all inside this folder.
# ./setup.sh --refine also downloads the image-editing model for the optional --refine pass
# (--refine=<model> for another one; see `env/bin/python lineart.py --help`).
set -euo pipefail
cd "$(dirname "$0")"

REFINE=""
for arg in "$@"; do
  case "$arg" in
    --refine) REFINE=default ;;
    --refine=*) REFINE="${arg#--refine=}" ;;
    *) echo "usage: ./setup.sh [--refine[=MODEL]]" >&2; exit 2 ;;
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

if [ -n "$REFINE" ]; then
  HF_HUB_OFFLINE=0 env/bin/python -W ignore - "$REFINE" <<'PY'
import sys
from rora import refine
key = refine.DEFAULT_MODEL if sys.argv[1] == "default" else sys.argv[1]
if key not in refine.MODELS:
    sys.exit(f"unknown model {key!r}; choose from {', '.join(refine.MODELS)}")
m = refine.MODELS[key]
print(f"Downloading {m.name} for --refine into the shared Hugging Face cache "
      f"(about {m.download_gb:.0f} GB, first time only)...")
refine.download(key)
PY
fi

echo
echo "Done. Put photos in img/in/ and run:"
echo "  env/bin/python lineart.py --style tattoo --preview"
if [ -z "$REFINE" ]; then
  echo
  echo "Optional: run ./setup.sh --refine to enable the --refine pass (downloads about 16 GB)."
fi
