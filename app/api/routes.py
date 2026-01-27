import io
import json
import zipfile
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
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

@router.post(
    "/infer/gradcam/zip",
    response_class=Response,
    responses={200: {"content": {"application/zip": {}}}},
)
async def infer_gradcam_zip(
    request: Request,
    image: UploadFile = File(...),
    method: str | None = Form(None),
    threshold: float | None = Form(None),
    username: str | None = Form(None),
):
    _ = (method, threshold, username)
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_rgb = np.array(img, dtype=np.uint8)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    service = request.app.state.inference_service
    zip_bytes = service.predict_with_gradcam_zip_bytes(img_rgb)
    headers = {"Content-Disposition": "attachment; filename=gradcam.zip"}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)
