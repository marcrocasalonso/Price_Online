"""Helpers tied to Streamlit session state."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

SESSION_DEFAULTS: Dict[str, Any] = {
    "mode_compare": "same",
    "last_upload_sig_main": None,
    "last_upload_sig_a": None,
    "last_upload_sig_b": None,
    "chat_history_agent": [],
    "last_report_md": "",
    "last_diffs": None,
    "last_highlights": "",
    "sel_sheet_main": "(todas)",
    "sel_sheet_a": "(todas)",
    "sel_sheet_b": "(todas)",
    "sel_sections": [],
    "v1_main": None,
    "v2_main": None,
    "v1_a": None,
    "v2_b": None,
    "filtered_csv_text": "",
    "chat_open": True,
    "v1_label_display": "",
    "v2_label_display": "",
}


def ensure_session_defaults() -> None:
    """Initialise ``st.session_state`` with the expected keys."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def file_signature(uploaded_file) -> Optional[str]:
    """Return a stable signature for the uploaded file."""
    if uploaded_file is None:
        return None
    try:
        size = uploaded_file.size
    except Exception:
        size = len(uploaded_file.getbuffer())
    return f"{uploaded_file.name}:{size}"


def reset_state() -> None:
    """Reset derived state after uploading a new file."""
    st.session_state["chat_history_agent"] = []
    st.session_state["last_report_md"] = ""
    st.session_state["last_diffs"] = None
    st.session_state["last_highlights"] = ""
    st.session_state["filtered_csv_text"] = ""
    st.session_state["v1_label_display"] = ""
    st.session_state["v2_label_display"] = ""


def get_dataframe_for_chat() -> pd.DataFrame:
    """Return the dataframe that should be used as context for the chat."""
    mode = st.session_state.get("mode_compare", "same")
    if mode == "same":
        df_single = st.session_state.get("df_struct_main")
        if df_single is not None:
            sel_sheet = st.session_state.get("sel_sheet_main", "(todas)")
            if sel_sheet and sel_sheet != "(todas)":
                df_single = df_single[df_single["Sheet"] == sel_sheet]
            return df_single
    df_a = st.session_state.get("df_struct_a")
    df_b = st.session_state.get("df_struct_b")
    if df_a is not None and df_b is not None:
        sel_sheet_a = st.session_state.get("sel_sheet_a", "(todas)")
        sel_sheet_b = st.session_state.get("sel_sheet_b", "(todas)")
        if sel_sheet_a and sel_sheet_a != "(todas)":
            df_a = df_a[df_a["Sheet"] == sel_sheet_a]
        if sel_sheet_b and sel_sheet_b != "(todas)":
            df_b = df_b[df_b["Sheet"] == sel_sheet_b]
        return pd.concat([df_a, df_b], ignore_index=True)
    return pd.DataFrame()
