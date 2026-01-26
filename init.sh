#!/usr/bin/env bash
set -euo pipefail

# ---------- FOLDERS ----------
mkdir -p app/api app/core app/ml app/services app/utils
mkdir -p models/convnext
mkdir -p configs
mkdir -p scripts
mkdir -p tests

# ---------- PYTHON PACKAGE FILES ----------
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/ml/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py

# ---------- APP FILES ----------
touch app/main.py
touch app/api/routes.py
touch app/api/schemas.py
touch app/core/config.py
touch app/core/logging.py
touch app/ml/model_loader.py
touch app/ml/preprocessing.py
touch app/ml/inference.py
touch app/ml/gradcam.py
touch app/services/inference_service.py
touch app/utils/image_utils.py

# ---------- CONFIGS ----------
cat > configs/concepts.yaml <<'YAML'
# Concept definitions used by the service
# index corresponds to output neuron position in the concept head
concepts:
  - name: rule_free_space
    index: 0
  - name: rule_cable_routing
    index: 1
  - name: rule_alignment
    index: 2
  - name: rule_covering
    index: 3
YAML

cat > configs/thresholds.json <<'JSON'
{
  "rule_free_space": 0.5,
  "rule_cable_routing": 0.5,
  "rule_alignment": 0.5,
  "rule_covering": 0.5
}
JSON

# ---------- MODELS (placeholders) ----------
touch models/convnext/backbone.safetensors
touch models/convnext/last_stage.pth
touch models/concept_head.pth

# ---------- SCRIPTS ----------
touch scripts/export_thresholds.py

# ---------- TESTS ----------
touch tests/test_inference.py
touch tests/test_api.py

# ---------- ROOT FILES ----------
cat > .env <<'ENV'
# Example environment variables
APP_ENV=dev
DEVICE=auto            # auto|cpu|cuda
IMG_SIZE=320
ALPHA_OVERLAY=0.45

# Paths
CONCEPTS_PATH=configs/concepts.yaml
THRESHOLDS_PATH=configs/thresholds.json
BACKBONE_SAFETENSORS=models/convnext/backbone.safetensors
LAST_STAGE_PTH=models/convnext/last_stage.pth
CONCEPT_HEAD_PTH=models/concept_head.pth
ENV

cat > .gitignore <<'GIT'
__pycache__/
*.pyc
*.pyo
*.pyd
*.swp
.DS_Store
.env
.venv/
venv/
dist/
build/
*.egg-info/
logs/
outputs/
GIT

cat > requirements.txt <<'REQ'
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
python-multipart==0.0.9
numpy==1.26.4
pandas==2.2.2
opencv-python==4.10.0.84
Pillow==10.3.0
torch>=2.2.0
torchvision>=0.17.0
timm==1.0.7
safetensors==0.4.3
PyYAML==6.0.1
REQ

cat > Dockerfile <<'DOCKER'
FROM python:3.11-slim

WORKDIR /app

# System deps for opencv-python (headless not used here) and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKER

cat > README.md <<'MD'
# cbm-fastapi

FastAPI inference service for CBM Concept Head + Grad-CAM montage.

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
