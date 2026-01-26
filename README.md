# cbm-fastapi

FastAPI inference service for CBM Concept Head + Grad-CAM montage.

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
