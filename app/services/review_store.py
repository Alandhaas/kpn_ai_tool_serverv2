from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

from app.core.config import settings


@dataclass
class ReviewEntry:
    uuid: str
    user: str
    probability: float
    created_at: str


def _date_dir(date_str: str) -> Path:
    return settings.reviews_dir / date_str


def _index_path(date_str: str) -> Path:
    return _date_dir(date_str) / "index.json"


def ensure_date_dir(date_str: str) -> Path:
    date_dir = _date_dir(date_str)
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir


def save_review_zip(date_str: str, review_uuid: str, zip_bytes: bytes) -> Path:
    date_dir = ensure_date_dir(date_str)
    zip_path = date_dir / f"{review_uuid}.zip"
    zip_path.write_bytes(zip_bytes)
    return zip_path


def load_review_zip(date_str: str, review_uuid: str) -> bytes | None:
    zip_path = _date_dir(date_str) / f"{review_uuid}.zip"
    if not zip_path.exists():
        return None
    return zip_path.read_bytes()


def list_reviews(date_str: str) -> List[Dict[str, Any]]:
    index_path = _index_path(date_str)
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    return []


def append_review_entry(date_str: str, entry: ReviewEntry) -> None:
    date_dir = ensure_date_dir(date_str)
    index_path = date_dir / "index.json"

    current = []
    if index_path.exists():
        try:
            current = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            current = []

    current.append(asdict(entry))
    index_path.write_text(json.dumps(current))
