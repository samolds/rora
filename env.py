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


def disable_ultralytics_analytics():
    from ultralytics import settings

    if settings.get("sync"):
        settings.update({"sync": False})
