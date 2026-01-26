import logging
import timm
import torch
from safetensors.torch import load_file

from app.core.config import settings
from app.ml.model_defs import ConvNeXtFeatures, ConceptHead

logger = logging.getLogger(__name__)


class ModelBundle:
    """
    Container voor alle ML-modellen.
    Wordt 1x aangemaakt bij app startup.
    """

    def __init__(self, backbone, feature_extractor, concept_head):
        self.backbone = backbone
        self.feature_extractor = feature_extractor
        self.concept_head = concept_head


def load_models() -> ModelBundle:
    device = settings.device_torch
    logger.info(f"Loading models on device: {device}")

    # =====================
    # Load ConvNeXt backbone
    # =====================
    backbone = timm.create_model("convnext_base", pretrained=False)

    logger.info(f"Loading backbone weights: {settings.backbone_path}")
    backbone.load_state_dict(
        load_file(str(settings.backbone_path)),
        strict=False,
    )

    # =====================
    # Load finetuned last stage
    # =====================
    logger.info(f"Loading last stage weights: {settings.last_stage_path}")
    last_stage_state = torch.load(
        settings.last_stage_path,
        map_location="cpu",
    )
    backbone.stages[3].load_state_dict(last_stage_state, strict=True)

    backbone.to(device)
    backbone.eval()

    # =====================
    # Feature extractor
    # =====================
    feature_extractor = ConvNeXtFeatures(backbone).to(device).eval()

    # =====================
    # Concept head
    # =====================
    concept_head = ConceptHead()
    logger.info(f"Loading concept head: {settings.concept_head_path}")
    concept_head.load_state_dict(
        torch.load(settings.concept_head_path, map_location="cpu"),
        strict=True,
    )
    concept_head.to(device).eval()

    logger.info("Models loaded successfully")

    return ModelBundle(
        backbone=backbone,
        feature_extractor=feature_extractor,
        concept_head=concept_head,
    )
