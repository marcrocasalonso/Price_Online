"""Main content rendering helpers."""
from __future__ import annotations

from io import StringIO
import pandas as pd
import streamlit as st

from ..analysis.comparator import diffs_to_df
from ..analysis.reporting import build_export_filename
from ..data.cleaning import is_empty_value
from .sidebar import ComparisonContext, UploadResult


def _variant_card(df_variant: pd.DataFrame, label: str, variant: str) -> None:
    market = df_variant["market"].dropna().iloc[0] if not df_variant.empty else "—"
    ingested = df_variant["ingested_at"].dropna().iloc[0] if not df_variant.empty else "—"
    brand = df_variant["variant_brand"].dropna().iloc[0] if not df_variant.empty else ""
    model = df_variant["variant_model"].dropna().iloc[0] if not df_variant.empty else ""
    st.markdown(f"<b>{label} · {variant}</b>", unsafe_allow_html=True)
    st.caption(f"{brand or ''} {model or ''}".strip())
    st.markdown(
        f"<span class='pill pill-green'>{market}</span> <span class='pill'>{ingested}</span>",
        unsafe_allow_html=True,
    )


def _render_tabs(df_filtered: pd.DataFrame) -> None:
    tab1, tab2, tab3 = st.tabs(["Informe", "Diferencias (CSV)", "Datos A/B"])
    with tab1:
        if st.session_state.get("last_report_md"):
            st.markdown(st.session_state["last_report_md"])
            if st.session_state.get("last_highlights"):
                st.subheader("Highlights")
                st.write(st.session_state["last_highlights"])
        else:
            st.info("Pulsa “Comparar y redactar informe” en el panel izquierdo (Configuración).")
    with tab2:
        diffs = st.session_state.get("last_diffs", None)
        if diffs:
            df_diffs = diffs_to_df(
                diffs,
                st.session_state.get("v1_label_display", "A"),
                st.session_state.get("v2_label_display", "B"),
            )
            st.dataframe(df_diffs, use_container_width=True)
        else:
            st.info("Genera el informe para ver las diferencias.")
    with tab3:
        filtered_csv_text = st.session_state.get("filtered_csv_text", "")
        if filtered_csv_text:
            df_show = pd.read_csv(StringIO(filtered_csv_text))
            st.dataframe(df_show, use_container_width=True)
        else:
            st.dataframe(df_filtered, use_container_width=True)


def render_main_view(mode: str, uploads: UploadResult, context: ComparisonContext) -> None:
    """Render the main column with variant details and tabs."""
    if mode == "same":
        if uploads.df_main is None or context.df_view_main is None:
            return
        export_name = uploads.export_name_main or build_export_filename(uploads.df_main)
        st.markdown(
            f"<b>Archivo generado:</b> {export_name} &nbsp; <span class='pill pill-navy'>{len(uploads.df_main)} filas</span>",
            unsafe_allow_html=True,
        )
        df_view = context.df_view_main
        variant_a = context.variant_a or "A"
        variant_b = context.variant_b or "B"
        col_a, col_b = st.columns(2)
        with col_a:
            _variant_card(df_view[df_view["Variant"] == variant_a], "A", variant_a)
        with col_b:
            _variant_card(df_view[df_view["Variant"] == variant_b], "B", variant_b)
        df_filtered = df_view[(df_view["Variant"] == variant_a) | (df_view["Variant"] == variant_b)].copy()
    else:
        if uploads.df_a is None or uploads.df_b is None:
            return
        export_name_a = uploads.export_name_a or build_export_filename(uploads.df_a)
        export_name_b = uploads.export_name_b or build_export_filename(uploads.df_b)
        st.markdown(
            f"<b>Excel A:</b> {export_name_a} &nbsp; <span class='pill pill-navy'>{len(uploads.df_a)} filas</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<b>Excel B:</b> {export_name_b} &nbsp; <span class='pill pill-navy'>{len(uploads.df_b)} filas</span>",
            unsafe_allow_html=True,
        )
        df_view_a = context.df_view_a
        df_view_b = context.df_view_b
        if df_view_a is None or df_view_b is None:
            return
        variant_a = context.variant_a or "A"
        variant_b = context.variant_b or "B"
        col_a, col_b = st.columns(2)
        with col_a:
            _variant_card(df_view_a[df_view_a["Variant"] == variant_a], "A", variant_a)
        with col_b:
            _variant_card(df_view_b[df_view_b["Variant"] == variant_b], "B", variant_b)
        df_filtered = pd.concat(
            [
                df_view_a[df_view_a["Variant"] == variant_a],
                df_view_b[df_view_b["Variant"] == variant_b],
            ],
            ignore_index=True,
        )
    df_filtered = df_filtered[~df_filtered["Value"].apply(is_empty_value)]
    _render_tabs(df_filtered)
