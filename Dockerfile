FROM python:3.12-slim

# --- system deps (opencv, PIL, torch need these)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --- env hygiene
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- workdir
WORKDIR /app

# --- install deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- copy code
COPY . .

# --- expose port
EXPOSE 8000

# --- start server (NO reload in Docker!)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
