"""Utilities for normalising and cleaning raw spreadsheet values."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

VALID_SECTIONS = [
    "General",
    "0. Basic Data",
    "1. Engine / Drive Train",
    "2. Chassis / Wheels",
    "3. Safety / Light",
    "4. Audio / Communication / Navigation",
    "5. Comfort",
    "6. Climate",
    "7. Car Body / Exterior",
    "8. Seats",
    "9. Interior",
    "10. Service / Tax",
    "Indices and interim values",
]


def norm_section_label(value: Optional[str]) -> str:
    """Return a normalised version of a section label."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    return text.replace("&", "and")


CANONICAL_BY_NORM = {norm_section_label(section): section for section in VALID_SECTIONS}
CANONICAL_SET_NORM = set(CANONICAL_BY_NORM.keys())
CANONICAL_SET = set(VALID_SECTIONS)
EMPTY_LIKE_RE = re.compile(
    r"^\s*(?:none|null|nan|n/?a|n\.a\.|—|-|–|\.|—\s*—)\s*$",
    re.IGNORECASE,
)


def clean_str(value: Optional[str]) -> Optional[str]:
    """Trim whitespace and collapse repeated spaces, returning ``None`` if empty."""
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else None


def is_empty_value(value: Optional[str]) -> bool:
    """Return ``True`` when ``value`` is considered empty for the UI."""
    if value is None:
        return True
    return bool(EMPTY_LIKE_RE.match(str(value)))


def now_ingested_at() -> str:
    """Timestamp helper used when exporting rows."""
    return datetime.now().isoformat(timespec="seconds")


def canonical_market(name: str) -> str:
    """Normalise market names to a canonical label."""
    if not name:
        return "Unknown"
    text = re.sub(r"\s+", " ", name).strip()
    title_cased = " ".join(word.capitalize() for word in text.split())
    mapping = {
        "Ger": "Germany",
        "De": "Germany",
        "Deu": "Germany",
        "Germany": "Germany",
        "Es": "Spain",
        "Esp": "Spain",
        "Spain": "Spain",
        "Fr": "France",
        "Fra": "France",
        "France": "France",
        "It": "Italy",
        "Ita": "Italy",
        "Italy": "Italy",
        "At": "Austria",
        "Aut": "Austria",
        "Austria": "Austria",
        "Pt": "Portugal",
        "Prt": "Portugal",
        "Portugal": "Portugal",
        "Uk": "United Kingdom",
        "Gbr": "United Kingdom",
        "Gb": "United Kingdom",
        "United Kingdom": "United Kingdom",
    }
    return mapping.get(title_cased, title_cased)


def parse_numeric_and_unit(value) -> Tuple[float, Optional[str]]:
    """Split a value into a numeric component and an optional unit."""
    if pd.isna(value) or value is None:
        return (np.nan, None)
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"yes", "y", "true", "si", "sí", "1"}:
        return (1.0, "bool")
    if lowered in {"no", "n", "false", "0"}:
        return (0.0, "bool")
    unit_guess = None
    for pattern in [
        r"kw",
        r"ps",
        r"hp",
        r"nm",
        r"km/h",
        r"wh/km",
        r"g/km",
        r"l/100km",
        r"€",
        r"eur",
        r"mm",
        r"cm",
        r"kg",
        r"inch",
        r"%",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            unit_guess = match.group(0).lower()
            break
    number_match = re.search(r"-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?", text)
    if not number_match:
        return (np.nan, unit_guess)
    num = number_match.group(0)
    if "," in num and "." in num:
        num = num.replace(".", "").replace(",", ".")
    elif "," in num and "." not in num:
        num = num.replace(",", ".")
    try:
        return (float(num), unit_guess)
    except Exception:
        return (np.nan, unit_guess)


def split_brand_model(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split a variant string into brand and model components."""
    if value is None:
        return (None, None)
    parts = [part.strip() for part in str(value).split("/")]
    if len(parts) >= 2:
        return parts[0], "/".join(parts[1:]).strip()
    tokens = str(value).split()
    if len(tokens) >= 2:
        return tokens[0], " ".join(tokens[1:])
    return (None, value)
