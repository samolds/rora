"""Keep every model, cache and setting inside this project folder. Import before torch/ultralytics."""
import logging
import os
from pathlib import Path

import truststore

# verify HTTPS with the OS certificate store, so downloads also work on networks that inspect
# HTTPS (e.g. Cisco Secure Access), whose root certificate Python's own bundle lacks
truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# the big --refine model lives in the user's shared Hugging Face cache (where mflux and other
# projects look too), resolved before HF_HOME is pointed at ./models below
SHARED_HF_HUB = Path(os.environ.get("RORA_REFINE_CACHE") or os.environ.get("HF_HUB_CACHE")
                     or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub").expanduser()

os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS / "ultralytics"))
# the xet transfer backend failed mid-download for the line-art weights; plain HTTP works
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# transformers imports torch.distributed.checkpoint, which logs this on every macOS import
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

HF_REPOS = ["lllyasviel/Annotators", "google/owlv2-base-patch16-ensemble",
            "depth-anything/Depth-Anything-V2-Small-hf"]


def _hf_cached(repo: str) -> bool:
    snapshots = MODELS / "hf" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.rglob("*.*"))


# once everything is downloaded, skip the per-run "is there a newer version?" checks;
# setup.sh passes HF_HUB_OFFLINE=0 so it can still download what's missing
if all(_hf_cached(r) for r in HF_REPOS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ultralytics resolves one.one.one.one / dns.google on import to decide whether it's online (and
# if so may check PyPI for updates); with its weights here it never needs the network.
# setup.sh passes YOLO_OFFLINE=0 so it can still download them
if all((MODELS / f).exists() for f in ("yolo11x.pt", "sam2.1_b.pt")):
    os.environ.setdefault("YOLO_OFFLINE", "1")


def disable_ultralytics_analytics():
    from ultralytics import settings

    if settings.get("sync"):
        settings.update({"sync": False})
