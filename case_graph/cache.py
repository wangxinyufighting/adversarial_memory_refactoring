import json
from pathlib import Path
from typing import Dict

from .models import ExtractionResult


CACHE_VERSION = 1


def load_extraction_cache(path: Path) -> Dict[str, ExtractionResult]:
    """Load persisted session extractions so reruns do not resend the same text."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != CACHE_VERSION:
        return {}
    items = payload.get("items", {})
    return {key: ExtractionResult.from_dict(value) for key, value in items.items()}


def save_extraction_cache(path: Path, cache: Dict[str, ExtractionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "items": {key: value.to_dict() for key, value in sorted(cache.items())},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
