"""Application configuration helpers."""
from __future__ import annotations

import os
from functools import lru_cache

import streamlit as st
from dotenv import load_dotenv

PAGE_TITLE = "POC (Price Online Comparator)"
PAGE_LAYOUT = "wide"


def setup_page() -> None:
    """Configure Streamlit global options for the app."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=None, layout=PAGE_LAYOUT)


@lru_cache(maxsize=1)
def load_environment() -> str:
    """Load environment variables and validate the OpenAI key.

    Returns the resolved ``OPENAI_API_KEY`` so that modules that depend on it
    can import :mod:`price_online.config` and trigger validation lazily.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error("Falta OPENAI_API_KEY (en .env o entorno).")
        raise SystemExit("OPENAI_API_KEY is required")
    return api_key
