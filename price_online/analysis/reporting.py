"""Helpers for building report artefacts."""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd


def build_export_filename(df: pd.DataFrame) -> str:
    """Build a deterministic export filename for a parsed dataframe."""
    reference_df = df[df["is_reference"] == True]  # noqa: E712 - pandas comparison
    model = None
    if not reference_df.empty:
        model = reference_df["variant_model"].dropna().iloc[0] or reference_df["Variant"].dropna().iloc[0]
    else:
        if "variant_model" in df.columns and df["variant_model"].notna().any():
            model = df["variant_model"].dropna().iloc[0]
        elif "Variant" in df.columns and df["Variant"].notna().any():
            model = df["Variant"].dropna().iloc[0]
    model_slug = re.sub(r"[^a-z0-9]+", "_", (model or "model").lower()).strip("_")
    market = df["market"].dropna().iloc[0] if "market" in df.columns and df["market"].notna().any() else "unknown"
    market_slug = re.sub(r"[^a-z0-9]+", "_", str(market).lower()).strip("_")
    timestamp = (
        df["ingested_at"].dropna().iloc[0]
        if "ingested_at" in df.columns and df["ingested_at"].notna().any()
        else datetime.now().isoformat(timespec="seconds")
    )
    return f"{model_slug}_{market_slug}_{timestamp}.csv"
