# agents/analysis/category_utils.py
"""
Tiny helper around categories.json
• category_group_of("Snack")   -> "Meals"
• merge_window_of("Meals")     -> 30 (minutes)
"""
from pathlib import Path
import json

# ------------------------------------------------------------
_BASE = Path(__file__).parent
_PATH = _BASE / "categories.json"

with open(_PATH, encoding="utf-8") as f:
    _RAW = json.load(f)

# parent → meta
# meta = { "merge_window_min": int, "items": [...] }
_SUB_TO_PARENT = {
    item: parent
    for parent, meta in _RAW.items()
    for item in meta["items"]
}

def category_group_of(label: str) -> str:
    return _SUB_TO_PARENT.get(label, "General")

def merge_window_of(group: str) -> int:
    return _RAW.get(group, _RAW["General"])["merge_window_min"]
