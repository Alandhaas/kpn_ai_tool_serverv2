import cv2
from pathlib import Path

from app.ml.model_loader import load_models
from app.ml.thresholds import load_thresholds
from app.services.inference_service import InferenceService


def test_single_image_inference_smoke():
    # --- load image
    img_path = Path(__file__).parent / "assets" / "image.JPG"
    assert img_path.exists(), "Test image missing"

    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, "cv2.imread failed"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- load models + thresholds
    models = load_models()
    thresholds = load_thresholds()

    service = InferenceService(
        models=models,
        thresholds=thresholds,
        concept_order=[
            "rule_free_space",
            "rule_cable_routing",
            "rule_alignment",
            "rule_covering",
        ],
    )

    out = service.predict(img_rgb)

    # --- assertions (minimal but meaningful)
    assert "final_ok" in out
    assert "per_concept" in out
    assert len(out["per_concept"]) == 4

    for concept, data in out["per_concept"].items():
        assert "p_ok" in data
        assert 0.0 <= data["p_ok"] <= 1.0
        assert "pred_ok" in data
