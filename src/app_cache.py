"""Shared Streamlit caches for data used across application pages."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_services import REVIEWS_PATH, load_reviews


@st.cache_data(
    persist="disk",
    max_entries=3,
    show_spinner="Đang tải dữ liệu review...",
)
def _load_reviews_cached(source: str, modified_ns: int) -> pd.DataFrame:
    """Load a versioned dataset; the timestamp invalidates stale disk entries."""
    del modified_ns
    return load_reviews(source)


def get_reviews(path: str | Path = REVIEWS_PATH) -> pd.DataFrame:
    """Return the shared review dataset cache for every Streamlit page."""
    source = Path(path)
    modified_ns = source.stat().st_mtime_ns if source.exists() else 0
    return _load_reviews_cached(str(source), modified_ns)
