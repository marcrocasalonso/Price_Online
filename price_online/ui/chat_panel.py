"""Chat panel rendering helpers."""
from __future__ import annotations

import streamlit as st

from ..chat.context import build_chat_context, render_context_as_text
from ..chat.service import llm_chat_answer, stream_openai_answer
from ..data.cleaning import is_empty_value
from . import state as ui_state


def render_chat_panel() -> None:
    """Render the chat panel on the right side of the layout."""
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.markdown('<div class="chat-head">Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-body">', unsafe_allow_html=True)
    for role, content in st.session_state.chat_history_agent:
        role_class = "msg user" if role == "user" else "msg"
        st.markdown(f"<div class='{role_class}'>{content}</div>", unsafe_allow_html=True)
    stream_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_area(
            "AI Agent",
            height=120,
            placeholder="Ej.: ¿Quiero saber los valores de ajuste que hacen subir o bajar el precio del coche B y cual es el valor de la suma total?",
        )
        submitted_chat = st.form_submit_button("Enviar")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if not (submitted_chat and user_msg and user_msg.strip()):
        return
    user_msg = user_msg.strip()
    st.session_state.chat_history_agent.append(("user", user_msg))
    df_source = ui_state.get_dataframe_for_chat()
    selected_sections = st.session_state.get("sel_sections", None)
    if selected_sections:
        df_source = df_source[df_source["Section"].isin(selected_sections)]
    if st.session_state.get("mode_compare", "same") == "same":
        variant_a = st.session_state.get("v1_main")
        variant_b = st.session_state.get("v2_main")
    else:
        variant_a = st.session_state.get("v1_a")
        variant_b = st.session_state.get("v2_b")
    filtered_csv_text = st.session_state.get("filtered_csv_text", "")
    if not filtered_csv_text and not df_source.empty:
        subset = df_source[df_source["Variant"].isin([variant_a, variant_b])].copy()
        subset = subset[~subset["Value"].apply(is_empty_value)]
        filtered_csv_text = subset.to_csv(index=False)
    report_md = st.session_state.get("last_report_md", "")
    if df_source.empty or not filtered_csv_text or variant_a is None or variant_b is None:
        answer = "No hay datos relevantes para responder."
        stream_placeholder.markdown(f"<div class='msg'>{answer}</div>", unsafe_allow_html=True)
    else:
        df_rag = build_chat_context(
            df_source,
            variant_a,
            variant_b,
            selected_sections,
            user_msg,
            top_k=160,
        )
        if df_rag.empty:
            answer = "No hay datos suficientes en el contexto actual (revisa hoja/secciones/variantes)."
            stream_placeholder.markdown(f"<div class='msg'>{answer}</div>", unsafe_allow_html=True)
        else:
            context_text = render_context_as_text(df_rag)
            try:
                answer = stream_openai_answer(
                    context_text=context_text,
                    user_q=user_msg,
                    filtered_csv_text=filtered_csv_text,
                    report_md=report_md,
                    stream_placeholder=stream_placeholder,
                    model="gpt-4o",
                    temperature=0.2,
                )
            except Exception:
                answer = llm_chat_answer(context_text, user_msg, filtered_csv_text, report_md)
                stream_placeholder.markdown(f"<div class='msg'>{answer}</div>", unsafe_allow_html=True)
    st.session_state.chat_history_agent.append(("assistant", answer))
    st.experimental_rerun()
