from pydantic import BaseModel
from typing import Dict

class ConceptDecision(BaseModel):
    p_ok: float
    threshold: float
    pred_ok: bool

class InferenceResponse(BaseModel):
    final_ok: bool
    per_concept: Dict[str, ConceptDecision]
    final_prob: float | None = None
    final_pred: bool | None = None
    final_threshold: float | None = None

class GradCamResponse(InferenceResponse):
    gradcam_png_base64: str
