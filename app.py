"""Streamlit entry-point orchestrating the Price Online Comparator."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from price_online import config, styles
from price_online.analysis import highlights
from price_online.analysis.comparator import compare_variants_by_section, render_differences_md
from price_online.ui import chat_panel as chat_ui
from price_online.ui import main_view as main_view_ui
from price_online.ui import sidebar as sidebar_ui
from price_online.ui import state as ui_state

# ----- Global setup -----
config.setup_page()
config.load_environment()
styles.inject_global_styles()
ui_state.ensure_session_defaults()

st.markdown("## Price Online Comparator (POC)")
st.caption("Digitaliza el Price Online, compara variantes y realiza preguntas.")

st.markdown('<div class="header-row">', unsafe_allow_html=True)
chat_toggle = st.toggle(
    "Mostrar AI Agent",
    value=st.session_state["chat_open"],
    help="Oculta o muestra el panel de agente IA",
)
st.markdown('</div>', unsafe_allow_html=True)
st.session_state["chat_open"] = chat_toggle

# ----- Sidebar orchestration -----
with st.sidebar:
    mode = sidebar_ui.render_mode_selector()
    uploads = sidebar_ui.render_uploaders(mode)
    comparison_context = sidebar_ui.render_configuration_panel(mode, uploads)
    sidebar_ui.render_download_buttons(mode, uploads, comparison_context)

# ----- Guard clauses when uploads are missing -----
if mode == "same" and uploads.df_main is None:
    st.info("Sube el Excel para continuar.")
    st.stop()
if mode == "cross" and (uploads.df_a is None or uploads.df_b is None):
    st.info("Sube ambos Excels (A y B) para continuar.")
    st.stop()

# ----- Report generation -----
if comparison_context.generate_report:
    selected_sections = st.session_state["sel_sections"] or None
    if mode == "same":
        df_view = comparison_context.df_view_main or uploads.df_main
        variant_a = comparison_context.variant_a
        variant_b = comparison_context.variant_b
        if df_view is not None and variant_a and variant_b:
            diffs = compare_variants_by_section(df_view, variant_a, variant_b, sections=selected_sections)
            sheet_main = comparison_context.sheet_main
            sheet_label = None if sheet_main in (None, "(todas)") else sheet_main
            report_md = render_differences_md(diffs, variant_a, variant_b, sheet_label)
            st.session_state["last_diffs"] = diffs
            st.session_state["last_report_md"] = report_md
            st.session_state["v1_label_display"] = variant_a
            st.session_state["v2_label_display"] = variant_b
            if diffs:
                st.session_state["last_highlights"] = highlights.generate_highlights(report_md)
            else:
                st.session_state["last_highlights"] = ""
    else:
        df_view_a = comparison_context.df_view_a or uploads.df_a
        df_view_b = comparison_context.df_view_b or uploads.df_b
        variant_a = comparison_context.variant_a
        variant_b = comparison_context.variant_b
        if df_view_a is not None and df_view_b is not None and variant_a and variant_b:
            df_a_sel = df_view_a[df_view_a["Variant"] == variant_a].copy()
            df_b_sel = df_view_b[df_view_b["Variant"] == variant_b].copy()
            variant_a_label = f"{variant_a} [A]"
            variant_b_label = f"{variant_b} [B]"
            df_a_sel["Variant"] = variant_a_label
            df_b_sel["Variant"] = variant_b_label
            combined = pd.concat([df_a_sel, df_b_sel], ignore_index=True)
            diffs = compare_variants_by_section(combined, variant_a_label, variant_b_label, sections=selected_sections)
            sheet_a = comparison_context.sheet_a
            sheet_b = comparison_context.sheet_b
            if sheet_a not in (None, "(todas)") or sheet_b not in (None, "(todas)"):
                sheet_label = f"A:{sheet_a} / B:{sheet_b}"
            else:
                sheet_label = None
            report_md = render_differences_md(diffs, variant_a_label, variant_b_label, sheet_label)
            st.session_state["last_diffs"] = diffs
            st.session_state["last_report_md"] = report_md
            st.session_state["v1_label_display"] = variant_a_label
            st.session_state["v2_label_display"] = variant_b_label
            if diffs:
                st.session_state["last_highlights"] = highlights.generate_highlights(report_md)
            else:
                st.session_state["last_highlights"] = ""

# ----- Layout and content -----
if st.session_state.get("chat_open", True):
    main_col, right_col = st.columns([0.90, 0.30], gap="large")
else:
    main_col = st.container()
    right_col = None

with main_col:
    main_view_ui.render_main_view(mode, uploads, comparison_context)

if chat_toggle and right_col is not None:
    with right_col:
        chat_ui.render_chat_panel()
