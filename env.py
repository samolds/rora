"""Keep every model, cache and setting inside this project folder. Import before torch/ultralytics."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS / "ultralytics"))
# the xet transfer backend failed mid-download for the line-art weights; plain HTTP works
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HF_REPOS = ["lllyasviel/Annotators", "google/owlv2-base-patch16-ensemble",
            "depth-anything/Depth-Anything-V2-Small-hf"]


def _hf_cached(repo: str) -> bool:
    snapshots = MODELS / "hf" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.rglob("*.*"))


# once everything is downloaded, skip the per-run "is there a newer version?" checks;
# setup.sh passes HF_HUB_OFFLINE=0 so it can still download what's missing
if all(_hf_cached(r) for r in HF_REPOS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def disable_ultralytics_analytics():
    from ultralytics import settings

    if settings.get("sync"):
        settings.update({"sync": False})
