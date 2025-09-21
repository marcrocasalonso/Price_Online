"""Parsing utilities for transforming Excel workbooks into structured data."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .cleaning import (
    CANONICAL_BY_NORM,
    CANONICAL_SET_NORM,
    clean_str,
    canonical_market,
    norm_section_label,
    now_ingested_at,
    parse_numeric_and_unit,
    split_brand_model,
)


def find_header_row_anywhere(df: pd.DataFrame) -> int:
    """Return the index of the row that contains the header information."""
    for idx in range(df.shape[0]):
        row_values = df.iloc[idx].astype(str).str.strip().str.lower().tolist()
        if "feature/attribute" in row_values:
            return idx
    counts = df.notna().sum(axis=1)
    return int(counts.idxmax())


def extract_market_from_first_row(raw: pd.DataFrame) -> str:
    """Infer the market from the first row of the sheet."""
    if raw.shape[0] == 0:
        return "Unknown"
    first_row_vals = raw.iloc[0].astype(str).fillna("").tolist()
    joined = " | ".join(first_row_vals)
    match = re.search(r"Adjustment\s*-\s*([^-–—|]+?)\s*-\s*", joined, flags=re.IGNORECASE)
    if match:
        return canonical_market(match.group(1))
    match = re.search(
        r"\b(germany|spain|france|italy|austria|portugal|united kingdom|uk)\b",
        joined,
        flags=re.IGNORECASE,
    )
    return canonical_market(match.group(1)) if match else "Unknown"


def parse_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """Parse an individual sheet and normalise it into a flat dataframe."""
    raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str)
    if raw.empty:
        return pd.DataFrame()
    market_sheet = extract_market_from_first_row(raw)
    raw = raw.dropna(how="all").dropna(how="all", axis=1).reset_index(drop=True)
    if raw.empty or raw.shape[1] < 3:
        return pd.DataFrame()
    header_row = find_header_row_anywhere(raw)
    header = list(raw.iloc[header_row, :])
    if len(header) >= 1:
        header[0] = "Feature/Attribute"
    if len(header) >= 2:
        header[1] = "Adjustment value"
    for col_idx in range(2, len(header)):
        value = header[col_idx]
        if pd.isna(value) or str(value).strip() == "" or str(value).lower() == "nan":
            header[col_idx] = header[col_idx - 1]
    data = raw.iloc[header_row + 1 :, : len(header)].copy()
    data.columns = header
    data = data.dropna(how="all").reset_index(drop=True)
    feature_attribute = data["Feature/Attribute"].apply(clean_str)
    feature_norm = feature_attribute.apply(norm_section_label)
    is_section = feature_norm.isin(CANONICAL_SET_NORM)
    data["Section"] = pd.Series(pd.NA, index=data.index, dtype="object")
    data.loc[is_section, "Section"] = feature_norm.loc[is_section].map(CANONICAL_BY_NORM)
    data["Section"] = data["Section"].ffill().fillna("summary")
    data = data[~is_section].copy().reset_index(drop=True)

    rows: List[Tuple] = []
    for col_idx in range(2, len(header)):
        base = clean_str(header[col_idx])
        if base is None or base.lower() == "adjustment value":
            continue
        sub_idx = 1
        for prev in range(2, col_idx):
            if clean_str(header[prev]) == base:
                sub_idx += 1
        series = data.iloc[:, col_idx]
        for row_i in range(len(data)):
            feat = clean_str(data.iloc[row_i, 0])
            sect = clean_str(data.loc[row_i, "Section"])
            val = clean_str(series.iloc[row_i])
            rows.append((market_sheet, sheet_name, sect, feat, base, sub_idx, val))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "market",
            "Sheet",
            "Section",
            "Feature/Attribute",
            "Variant",
            "sub_idx",
            "Value",
        ],
    )
    parsed = df["Value"].apply(parse_numeric_and_unit)
    df["value_num"] = [item[0] for item in parsed]
    df["unit_guess"] = [item[1] for item in parsed]
    brands_models = df["Variant"].apply(split_brand_model)
    df["variant_brand"] = [item[0] for item in brands_models]
    df["variant_model"] = [item[1] for item in brands_models]
    df["is_reference"] = df["variant_brand"].str.upper().isin(["SEAT", "CUPRA"])
    df["ingested_at"] = now_ingested_at()
    df = df[
        [
            "market",
            "Sheet",
            "Section",
            "Feature/Attribute",
            "Variant",
            "variant_brand",
            "variant_model",
            "sub_idx",
            "is_reference",
            "Value",
            "value_num",
            "unit_guess",
            "ingested_at",
        ]
    ]
    return df


def build_csv_from_excel(xlsx_path: str, out_csv_path: str) -> pd.DataFrame:
    """Parse an Excel workbook and export the normalised data to CSV."""
    xls = pd.ExcelFile(xlsx_path)

    def norm_simple(value: str) -> str:
        return re.sub(r"[\s_-]+", "", str(value).lower())

    skip = {name for name in xls.sheet_names if norm_simple(name) in {"indexoverview", "additions"}}
    frames: List[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        if sheet in skip:
            continue
        try:
            df_sheet = parse_sheet(xls, sheet)
            if not df_sheet.empty:
                frames.append(df_sheet)
        except Exception as exc:  # pragma: no cover - defensive path
            frames.append(
                pd.DataFrame(
                    {
                        "market": ["Unknown"],
                        "Sheet": [sheet],
                        "Section": ["summary"],
                        "Feature/Attribute": [f"ERROR: {exc}"],
                        "Variant": [None],
                        "variant_brand": [None],
                        "variant_model": [None],
                        "sub_idx": [np.nan],
                        "is_reference": [False],
                        "Value": [None],
                        "value_num": [np.nan],
                        "unit_guess": [None],
                        "ingested_at": [now_ingested_at()],
                    }
                )
            )
    if frames:
        output = pd.concat(frames, ignore_index=True)
    else:
        output = pd.DataFrame(
            columns=[
                "market",
                "Sheet",
                "Section",
                "Feature/Attribute",
                "Variant",
                "variant_brand",
                "variant_model",
                "sub_idx",
                "is_reference",
                "Value",
                "value_num",
                "unit_guess",
                "ingested_at",
            ]
        )
    Path(out_csv_path).write_bytes(output.to_csv(index=False).encode("utf-8"))
    return output
