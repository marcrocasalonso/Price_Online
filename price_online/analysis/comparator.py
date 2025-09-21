"""Variant comparison utilities."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..data.cleaning import VALID_SECTIONS, clean_str, is_empty_value


def list_variants(df: pd.DataFrame) -> List[str]:
    """Return the sorted list of variants present in ``df``."""
    if "Variant" not in df.columns:
        return []
    return sorted([variant for variant in df["Variant"].dropna().unique()])


def sections_in_df(df: pd.DataFrame) -> List[str]:
    """Return the sections appearing in ``df`` ordered like ``VALID_SECTIONS``."""
    if "Section" not in df.columns:
        return []
    sections = [section for section in df["Section"].dropna().unique()]
    order = {section: idx for idx, section in enumerate(VALID_SECTIONS)}
    return sorted(sections, key=lambda section: order.get(section, 999))


def _collapse_two_values(group: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    group = group.sort_values("sub_idx", kind="mergesort")
    value_1 = clean_str(group["Value"].iloc[0]) if len(group) >= 1 else None
    value_2 = clean_str(group["Value"].iloc[1]) if len(group) >= 2 else None
    value_1 = None if is_empty_value(value_1) else value_1
    value_2 = None if is_empty_value(value_2) else value_2
    return value_1, value_2


def build_variant_matrix(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Return a matrix with at most two values per feature for ``variant``."""
    subset = df[df["Variant"] == variant][["Section", "Feature/Attribute", "sub_idx", "Value"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["Section", "Feature/Attribute", "value_1", "value_2"])
    rows = []
    for (section, feature), group in subset.groupby(["Section", "Feature/Attribute"], dropna=True):
        value_1, value_2 = _collapse_two_values(group)
        if value_1 is None and value_2 is not None:
            value_1, value_2 = value_2, None
        if value_1 is None and value_2 is None:
            continue
        rows.append((section, feature, value_1, value_2))
    return pd.DataFrame(rows, columns=["Section", "Feature/Attribute", "value_1", "value_2"]).drop_duplicates()


def compare_variants_by_section(
    df: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    sections: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return the per-section differences between ``variant_a`` and ``variant_b``."""
    selected_sections = sections if sections else sections_in_df(df)
    matrix_a = build_variant_matrix(df, variant_a).set_index(["Section", "Feature/Attribute"])
    matrix_b = build_variant_matrix(df, variant_b).set_index(["Section", "Feature/Attribute"])
    if matrix_a.empty or matrix_b.empty:
        return {}
    differences: Dict[str, List[Tuple[str, str, str]]] = {}
    common_idx = matrix_a.index.intersection(matrix_b.index)
    for section in selected_sections:
        features = [idx for idx in common_idx if idx[0] == section]
        if not features:
            continue
        section_diffs: List[Tuple[str, str, str]] = []
        for _, feature in features:
            values_a = " | ".join(
                [value for value in [matrix_a.at[(section, feature), "value_1"], matrix_a.at[(section, feature), "value_2"]] if value]
            )
            values_b = " | ".join(
                [value for value in [matrix_b.at[(section, feature), "value_1"], matrix_b.at[(section, feature), "value_2"]] if value]
            )
            if values_a != values_b:
                section_diffs.append((feature, values_a if values_a else "—", values_b if values_b else "—"))
        if section_diffs:
            differences[section] = section_diffs
    return differences


def render_differences_md(
    diffs: Dict[str, List[Tuple[str, str, str]]],
    variant_a: str,
    variant_b: str,
    sheet: Optional[str] = None,
) -> str:
    """Render the ``diffs`` dictionary as Markdown."""
    if not diffs:
        text = f"# Informe de diferencias\n\nNo se han encontrado diferencias entre {variant_a} y {variant_b}."
        if sheet:
            text += f"\n\nHoja: {sheet}"
        return text
    lines = [f"# Informe de diferencias\n\nA: {variant_a}\n\nB: {variant_b}\n"]
    if sheet:
        lines.append(f"Hoja: {sheet}")
    for section, items in diffs.items():
        lines.append(f"\n## {section}")
        for feature, value_a, value_b in items:
            lines.append(f"- {feature}\n  - A: {value_a}\n  - B: {value_b}")
    return "\n".join(lines)


def diffs_to_df(diffs: Dict[str, List[Tuple[str, str, str]]], variant_a: str, variant_b: str) -> pd.DataFrame:
    """Convert the diff dictionary into a dataframe representation."""
    rows = []
    for section, items in diffs.items():
        for feature, value_a, value_b in items:
            rows.append(
                {
                    "Section": section,
                    "Feature/Attribute": feature,
                    "A_Variant": variant_a,
                    "A_Value": value_a,
                    "B_Variant": variant_b,
                    "B_Value": value_b,
                }
            )
    return pd.DataFrame(rows)
