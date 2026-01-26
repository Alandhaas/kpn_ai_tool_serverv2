import io
import json
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import Response

from app.api.schemas import InferenceResponse, GradCamResponse

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/infer", response_model=InferenceResponse)
async def infer(request: Request, file: UploadFile = File(...)):
    # Basic content-type validation
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # Decode image safely
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_rgb = np.array(img, dtype=np.uint8)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Get service from app state
    service = request.app.state.inference_service
    out = service.predict(img_rgb)

    return out

@router.post("/infer/gradcam")
async def infer_gradcam(request: Request, file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_rgb = np.array(img, dtype=np.uint8)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    service = request.app.state.inference_service
    meta, png_bytes = service.predict_with_gradcam_bytes(img_rgb)

    # Optie: meta meegeven in header (frontend kan dit lezen)
    headers = {
        "X-Inference-Meta": json.dumps(meta)  # let op: headers hebben size limiet
    }

    return Response(content=png_bytes, media_type="image/png", headers=headers)
