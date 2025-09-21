"""Sidebar rendering helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from ..analysis.comparator import list_variants, sections_in_df
from ..analysis.reporting import build_export_filename
from ..data.cleaning import VALID_SECTIONS, is_empty_value
from ..data.parser import build_csv_from_excel
from . import state as ui_state


@dataclass
class UploadResult:
    df_main: Optional[pd.DataFrame] = None
    df_a: Optional[pd.DataFrame] = None
    df_b: Optional[pd.DataFrame] = None
    export_name_main: Optional[str] = None
    export_name_a: Optional[str] = None
    export_name_b: Optional[str] = None


@dataclass
class ComparisonContext:
    generate_report: bool = False
    df_view_main: Optional[pd.DataFrame] = None
    df_view_a: Optional[pd.DataFrame] = None
    df_view_b: Optional[pd.DataFrame] = None
    variant_a: Optional[str] = None
    variant_b: Optional[str] = None
    sheet_main: Optional[str] = None
    sheet_a: Optional[str] = None
    sheet_b: Optional[str] = None
    filtered_ab: Optional[pd.DataFrame] = None


def render_mode_selector() -> str:
    """Render the radio selector that switches comparison mode."""
    st.subheader("Modo de comparación")
    mode_label = st.radio(
        "Elige modo",
        ["Un solo Excel", "Dos Excels"],
        horizontal=False,
        index=0 if st.session_state["mode_compare"] == "same" else 1,
    )
    mode = "same" if mode_label == "Un solo Excel" else "cross"
    st.session_state["mode_compare"] = mode
    return mode


def render_uploaders(mode: str) -> UploadResult:
    """Render file uploaders according to ``mode`` and parse workbooks."""
    st.subheader("Cargar Excel")
    if mode == "same":
        xlsx_main = st.file_uploader("Excel (único)", type=["xlsx"], key="uploader_main")
        current_sig = ui_state.file_signature(xlsx_main)
        if current_sig != st.session_state["last_upload_sig_main"]:
            st.session_state["last_upload_sig_main"] = current_sig
            ui_state.reset_state()
        if not xlsx_main:
            return UploadResult()
        with st.spinner("Procesando Excel…"):
            tmp_xlsx_path = Path("uploaded_main.xlsx")
            tmp_xlsx_path.write_bytes(xlsx_main.getbuffer())
            out_csv_path = Path("specs_basic_flat_main.csv")
            df_struct_main = build_csv_from_excel(str(tmp_xlsx_path), str(out_csv_path))
        if df_struct_main.empty:
            st.error("No se pudo extraer información del Excel.")
            st.stop()
        export_name = build_export_filename(df_struct_main)
        st.session_state["df_struct_main"] = df_struct_main
        return UploadResult(df_main=df_struct_main, export_name_main=export_name)
    xlsx_a = st.file_uploader("Excel A", type=["xlsx"], key="uploader_a")
    xlsx_b = st.file_uploader("Excel B", type=["xlsx"], key="uploader_b")
    sig_a = ui_state.file_signature(xlsx_a)
    sig_b = ui_state.file_signature(xlsx_b)
    if sig_a != st.session_state["last_upload_sig_a"] or sig_b != st.session_state["last_upload_sig_b"]:
        st.session_state["last_upload_sig_a"] = sig_a
        st.session_state["last_upload_sig_b"] = sig_b
        ui_state.reset_state()
    if not (xlsx_a and xlsx_b):
        return UploadResult()
    with st.spinner("Procesando Excel A…"):
        tmp_path_a = Path("uploaded_a.xlsx")
        tmp_path_a.write_bytes(xlsx_a.getbuffer())
        out_csv_a = Path("specs_basic_flat_a.csv")
        df_struct_a = build_csv_from_excel(str(tmp_path_a), str(out_csv_a))
    with st.spinner("Procesando Excel B…"):
        tmp_path_b = Path("uploaded_b.xlsx")
        tmp_path_b.write_bytes(xlsx_b.getbuffer())
        out_csv_b = Path("specs_basic_flat_b.csv")
        df_struct_b = build_csv_from_excel(str(tmp_path_b), str(out_csv_b))
    if df_struct_a.empty or df_struct_b.empty:
        st.error("No se pudo extraer información de alguno de los Excels.")
        st.stop()
    export_name_a = build_export_filename(df_struct_a)
    export_name_b = build_export_filename(df_struct_b)
    st.session_state["df_struct_a"] = df_struct_a
    st.session_state["df_struct_b"] = df_struct_b
    return UploadResult(
        df_a=df_struct_a,
        df_b=df_struct_b,
        export_name_a=export_name_a,
        export_name_b=export_name_b,
    )


def _safe_index(options, value, fallback_idx: int) -> int:
    if not options:
        return 0
    return options.index(value) if (value in options) else min(fallback_idx, max(0, len(options) - 1))


def render_configuration_panel(mode: str, uploads: UploadResult) -> ComparisonContext:
    """Render the configuration controls and return the resulting selection."""
    st.divider()
    generate_report = False
    context = ComparisonContext(generate_report=False)
    with st.expander("Configuración", expanded=True):
        if mode == "same":
            df_struct_main = uploads.df_main
            if df_struct_main is None:
                return context
            sheets = sorted(df_struct_main["Sheet"].dropna().unique().tolist())
            options = ["(todas)"] + sheets
            selected_sheet = st.selectbox(
                "Hoja (único Excel)",
                options,
                index=options.index(st.session_state["sel_sheet_main"]) if st.session_state["sel_sheet_main"] in options else 0,
            )
            st.session_state["sel_sheet_main"] = selected_sheet
            sections_available = sections_in_df(df_struct_main)
            default_sections = (
                sections_available if not st.session_state["sel_sections"] else st.session_state["sel_sections"]
            )
            selected_sections = st.multiselect("Secciones", sections_available, default=default_sections)
            st.session_state["sel_sections"] = selected_sections
            df_view = (
                df_struct_main
                if selected_sheet == "(todas)"
                else df_struct_main[df_struct_main["Sheet"] == selected_sheet]
            )
            variants = list_variants(df_view)
            if len(variants) < 2:
                st.warning("No hay suficientes variantes en la hoja seleccionada.")
            idx_v1 = _safe_index(variants, st.session_state["v1_main"], 0)
            idx_v2 = _safe_index(variants, st.session_state["v2_main"], 1 if len(variants) > 1 else 0)
            variant_a = st.selectbox("Variante A", variants, index=idx_v1 if variants else 0)
            variant_b = st.selectbox("Variante B", variants, index=idx_v2 if variants else 0)
            st.session_state["v1_main"] = variant_a
            st.session_state["v2_main"] = variant_b
            filtered_sidebar = df_view[(df_view["Variant"] == variant_a) | (df_view["Variant"] == variant_b)].copy()
            filtered_sidebar = filtered_sidebar[~filtered_sidebar["Value"].apply(is_empty_value)]
            st.session_state["filtered_csv_text"] = (
                filtered_sidebar.to_csv(index=False) if not filtered_sidebar.empty else ""
            )
            st.markdown("---")
            generate_report = st.button("Comparar y redactar informe", use_container_width=True)
            context = ComparisonContext(
                generate_report=generate_report,
                df_view_main=df_view,
                variant_a=variant_a,
                variant_b=variant_b,
                sheet_main=selected_sheet,
            )
        else:
            df_struct_a = uploads.df_a
            df_struct_b = uploads.df_b
            if df_struct_a is None or df_struct_b is None:
                return context
            sheets_a = sorted(df_struct_a["Sheet"].dropna().unique().tolist())
            sheets_b = sorted(df_struct_b["Sheet"].dropna().unique().tolist())
            options_a = ["(todas)"] + sheets_a
            options_b = ["(todas)"] + sheets_b
            selected_sheet_a = st.selectbox(
                "Hoja Excel A",
                options_a,
                index=options_a.index(st.session_state["sel_sheet_a"]) if st.session_state["sel_sheet_a"] in options_a else 0,
            )
            selected_sheet_b = st.selectbox(
                "Hoja Excel B",
                options_b,
                index=options_b.index(st.session_state["sel_sheet_b"]) if st.session_state["sel_sheet_b"] in options_b else 0,
            )
            st.session_state["sel_sheet_a"] = selected_sheet_a
            st.session_state["sel_sheet_b"] = selected_sheet_b
            sections_a = set(sections_in_df(df_struct_a))
            sections_b = set(sections_in_df(df_struct_b))
            section_pool = sections_a | sections_b
            sections_available = sorted(
                section_pool,
                key=lambda section: VALID_SECTIONS.index(section) if section in VALID_SECTIONS else 999,
            )
            default_sections = (
                sections_available if not st.session_state["sel_sections"] else st.session_state["sel_sections"]
            )
            selected_sections = st.multiselect("Secciones", sections_available, default=default_sections)
            st.session_state["sel_sections"] = selected_sections
            df_view_a = (
                df_struct_a
                if selected_sheet_a == "(todas)"
                else df_struct_a[df_struct_a["Sheet"] == selected_sheet_a]
            )
            df_view_b = (
                df_struct_b
                if selected_sheet_b == "(todas)"
                else df_struct_b[df_struct_b["Sheet"] == selected_sheet_b]
            )
            variants_a = list_variants(df_view_a)
            variants_b = list_variants(df_view_b)
            idx_v1 = _safe_index(variants_a, st.session_state["v1_a"], 0)
            idx_v2 = _safe_index(variants_b, st.session_state["v2_b"], 0)
            variant_a = st.selectbox("Variante A (Excel A)", variants_a, index=idx_v1 if variants_a else 0)
            variant_b = st.selectbox("Variante B (Excel B)", variants_b, index=idx_v2 if variants_b else 0)
            st.session_state["v1_a"] = variant_a
            st.session_state["v2_b"] = variant_b
            df_a_sel = df_view_a[df_view_a["Variant"] == variant_a].copy()
            df_b_sel = df_view_b[df_view_b["Variant"] == variant_b].copy()
            df_combined = pd.concat([df_a_sel, df_b_sel], ignore_index=True)
            df_combined_ctx = df_combined[~df_combined["Value"].apply(is_empty_value)]
            st.session_state["filtered_csv_text"] = (
                df_combined_ctx.to_csv(index=False) if not df_combined_ctx.empty else ""
            )
            st.markdown("---")
            generate_report = st.button("Comparar y redactar informe", use_container_width=True)
            context = ComparisonContext(
                generate_report=generate_report,
                df_view_a=df_view_a,
                df_view_b=df_view_b,
                variant_a=variant_a,
                variant_b=variant_b,
                sheet_a=selected_sheet_a,
                sheet_b=selected_sheet_b,
                filtered_ab=df_combined_ctx,
            )
    return context


def render_download_buttons(mode: str, uploads: UploadResult, context: ComparisonContext) -> None:
    """Render download buttons depending on the current mode."""
    if mode == "same":
        df_struct_main = uploads.df_main
        if df_struct_main is None:
            return
        export_name = uploads.export_name_main or build_export_filename(df_struct_main)
        st.download_button(
            "Descargar CSV estructurado",
            data=df_struct_main.to_csv(index=False).encode("utf-8"),
            file_name=export_name,
            mime="text/csv",
            use_container_width=True,
            key="dl_structured_main",
        )
        filtered_text = st.session_state.get("filtered_csv_text", "")
        if filtered_text:
            st.download_button(
                "Descargar CSV filtrado (A/B)",
                data=filtered_text.encode("utf-8"),
                file_name=f"filtered_{export_name}",
                mime="text/csv",
                use_container_width=True,
                key="dl_filtered_main",
            )
        return
    df_struct_a = uploads.df_a
    df_struct_b = uploads.df_b
    if df_struct_a is None or df_struct_b is None:
        return
    export_name_a = uploads.export_name_a or build_export_filename(df_struct_a)
    export_name_b = uploads.export_name_b or build_export_filename(df_struct_b)
    st.download_button(
        "Descargar CSV A (estructurado)",
        data=df_struct_a.to_csv(index=False).encode("utf-8"),
        file_name=export_name_a,
        mime="text/csv",
        use_container_width=True,
        key="dl_structured_a",
    )
    st.download_button(
        "Descargar CSV B (estructurado)",
        data=df_struct_b.to_csv(index=False).encode("utf-8"),
        file_name=export_name_b,
        mime="text/csv",
        use_container_width=True,
        key="dl_structured_b",
    )
    if context.filtered_ab is not None and not context.filtered_ab.empty:
        st.download_button(
            "Descargar CSV filtrado (A+B)",
            data=context.filtered_ab.to_csv(index=False).encode("utf-8"),
            file_name=f"ab_{export_name_a.replace('.csv','')}__{export_name_b}",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_filtered_ab_{context.sheet_a}_{context.sheet_b}_{context.variant_a}_{context.variant_b}",
        )
