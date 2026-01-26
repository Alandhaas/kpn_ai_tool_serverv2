import json
from typing import Dict
from app.core.config import settings


def load_thresholds() -> Dict[str, float]:
    """
    Load production thresholds from configs/thresholds.json.

    Expected format:
    {
      "rule_free_space": float,
      "rule_cable_routing": float,
      "rule_alignment": float,
      "rule_covering": float,
      "threshold_epoch": int (optional)
    }
    """
    with open(settings.thresholds_path, "r") as f:
        data = json.load(f)

    thresholds = {}
    for key, value in data.items():
        if key.startswith("rule_"):
            if value is None or value != value:  # NaN check
                raise ValueError(f"Invalid threshold for {key}")
            thresholds[key] = float(value)

    if not thresholds:
        raise ValueError("No rule_* thresholds found in thresholds.json")

    return thresholds
