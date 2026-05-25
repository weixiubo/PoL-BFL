"""Minimal manifest writer for experiment runs.
Write a manifest.json under the given run_dir including config snapshot and helpful indexes.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

def write_manifest(run_dir: str | Path, config: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> Path:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    manifest = {
        'run_dir': str(p.resolve()),
        'config': config,
        'summary': summary or {},
    }
    out = p / 'manifest.json'
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return out

