"""Context building helpers for the chat assistant."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pandas as pd

from ..data.cleaning import is_empty_value


def build_chat_context(
    df: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    sections: Optional[Iterable[str]],
    user_question: str,
    top_k: int = 160,
) -> pd.DataFrame:
    """Return the most relevant rows to answer ``user_question``."""
    context = df[(df["Variant"] == variant_a) | (df["Variant"] == variant_b)].copy()
    if sections:
        context = context[context["Section"].isin(sections)]
    if context.empty:
        return context
    context = context[~context["Value"].apply(is_empty_value)].copy()
    tokens = [token for token in re.split(r"[^\w%/\.]+", user_question.lower()) if token]

    def _score_row(row: pd.Series) -> int:
        text = " ".join(
            [str(row.get(column) or "") for column in ["Section", "Feature/Attribute", "Variant", "Value"]]
        ).lower()
        return sum(1 for token in tokens if token in text)

    context["__score"] = context.apply(_score_row, axis=1)
    return context.sort_values(
        ["__score", "Section", "Feature/Attribute", "Variant", "sub_idx"],
        ascending=[False, True, True, True, True],
    ).head(top_k)


def render_context_as_text(df_context: pd.DataFrame) -> str:
    """Render ``df_context`` as plain text for prompting."""
    lines: List[str] = []
    for _, row in df_context.iterrows():
        sub = int(row["sub_idx"]) if pd.notna(row["sub_idx"]) else 1
        lines.append(f"[{row['Section']}] {row['Feature/Attribute']} — {row['Variant']} (sub{sub}): {row['Value']}")
    return "\n".join(lines)
