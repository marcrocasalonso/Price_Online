"""Styling utilities for the Streamlit UI."""
from __future__ import annotations

import streamlit as st

_GLOBAL_CSS = """
<style>
:root{
  --navy:#0b2b4c;
  --gray-900:#111827;
  --gray-700:#374151;
  --gray-500:#6b7280;
  --gray-300:#d1d5db;
  --gray-200:#e5e7eb;
  --gray-100:#f3f4f6;  /* gris sidebar */
  --white:#ffffff;
  --green:#2e7d32;
}

/* Tipografía */
html, body, [class*="css"]  {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  font-size:10px;
  color:var(--gray-900);
}
h1 { font-size:20px; font-weight:700; color:var(--navy); }
h2 { font-size:14px; font-weight:700; color:var(--navy); }
h3 { font-size:12px; font-weight:600; color:var(--navy); }
small { font-size:10px; color:var(--gray-500); }

/* Sidebar izquierdo (nativo) */
section[data-testid="stSidebar"] > div {
  width: 280px;
  min-width: 280px;
}

/* Tarjetas */
.card {
  border: 1px solid var(--gray-200);
  border-radius: 12px; padding: 10px 12px;
  background: var(--white); box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  margin-bottom:10px;
}
.pill {
  display:inline-block; padding:2px 8px; border-radius:999px;
  background: var(--gray-100); font-size:11px; font-weight:600; margin-right:6px; color:var(--gray-700);
}
.pill-green { background:#e8f5e9; color:var(--green); }
.pill-navy  { background:#e8eef6; color:var(--navy); }

/* Tabs */
.stTabs [role="tablist"] button[role="tab"] {
  font-size:13px;
}

/* ===== Chat tipo sidebar derecho (scroll independiente) ===== */
.chat-panel{
  position: sticky;
  top: 12px;
  display:flex; flex-direction:column;
  background: var(--gray-100);
  border: 1px solid var(--gray-200);
  border-radius: 12px;
  height: calc(100vh - 24px);
  overflow: hidden;
}
.chat-head{
  padding:10px 12px;
  border-bottom:1px solid var(--gray-200);
  background:var(--gray-100);
  font-weight:600;
}
.chat-body{
  flex:1;
  overflow:auto;               /* scroll independiente */
  padding:10px 12px;
}
.chat-input{
  border-top:1px solid var(--gray-200);
  padding:8px 10px;
  background:var(--gray-100);
}

/* Mensajes muy sencillos */
.msg{
  padding:8px 10px; border-radius:8px; margin:6px 0;
  border:1px solid var(--gray-200); background:var(--white);
  word-wrap:break-word; overflow-wrap:anywhere;
}
.msg.user{
  background:var(--navy); color:white; border-color:var(--navy);
}

/* Toggle fila superior (alinear a la derecha) */
.header-row{
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:8px;
}
</style>
"""


def inject_global_styles() -> None:
    """Inject the application CSS into the Streamlit page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
