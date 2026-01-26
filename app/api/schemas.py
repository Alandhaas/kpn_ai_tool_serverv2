from pydantic import BaseModel
from typing import Dict

class ConceptDecision(BaseModel):
    p_ok: float
    threshold: float
    pred_ok: bool

class InferenceResponse(BaseModel):
    final_ok: bool
    per_concept: Dict[str, ConceptDecision]

class GradCamResponse(InferenceResponse):
    gradcam_png_base64: str
