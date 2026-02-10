import io
import json
import zipfile
from datetime import datetime, timezone, date
from uuid import uuid4
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import Response

from app.api.schemas import InferenceResponse
from app.services.review_store import (
    ReviewEntry,
    append_review_entry,
    list_reviews,
    load_review_zip,
    save_review_zip,
)

router = APIRouter()

def _parse_date(date_str: str | None) -> str:
    if not date_str:
        return datetime.now(timezone.utc).date().isoformat()
    try:
        date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format (YYYY-MM-DD)")
    return date_str

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/infer", response_model=InferenceResponse)
async def infer(
    request: Request,
    file: UploadFile = File(...),
):
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
async def infer_gradcam(
    request: Request,
    file: UploadFile = File(...),
):
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

    review_uuid = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    date_str = created_at.split("T")[0]
    probability = 0.0
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
            meta_raw = zf.read("metadata.json")
            meta = json.loads(meta_raw)
            if isinstance(meta, list) and meta:
                probability = float(meta[0].get("final_prob", 0.0))
    except Exception:
        probability = 0.0

    entry = ReviewEntry(
        uuid=review_uuid,
        user=(username or "anonymous").strip() or "anonymous",
        probability=probability,
        created_at=created_at,
    )
    save_review_zip(date_str, review_uuid, zip_bytes)
    append_review_entry(date_str, entry)

    headers = {"Content-Disposition": "attachment; filename=gradcam.zip"}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


@router.get("/review/list")
async def review_list(date: str | None = None):
    date_str = _parse_date(date)
    scans = list_reviews(date_str)
    return {"date": date_str, "scans": scans}


@router.get(
    "/review/zip/{review_uuid}",
    response_class=Response,
    responses={200: {"content": {"application/zip": {}}}},
)
async def review_zip(review_uuid: str, date: str | None = None):
    date_str = _parse_date(date)
    zip_bytes = load_review_zip(date_str, review_uuid)
    if not zip_bytes:
        raise HTTPException(status_code=404, detail="Review not found")
    headers = {"Content-Disposition": f"attachment; filename={review_uuid}.zip"}
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)
