"""
dashboard/core/ui_helpers.py

Composants Streamlit partagés entre plusieurs pages du dashboard.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def truncate_label(label: str, max_len: int = 55) -> str:
    """
    Tronque un label trop long en ajoutant une ellipse.

    Args:
        label   : Texte à tronquer.
        max_len : Longueur maximale avant troncature (défaut : 55).

    Returns:
        Label original si inférieur à max_len, sinon label[:max_len] + "…".
    """
    if len(label) > max_len:
        return label[:max_len] + "…"
    return label


def render_category_selector(categories: list[tuple[str, str]]) -> Optional[str]:
    """
    Sélecteur de catégorie partagé entre les pages.

    Persiste la sélection dans session_state['selected_category'] via le
    paramètre key= de Streamlit.  Les pages suivantes peuvent lire
    session_state['selected_category'] directement.

    Args:
        categories : Liste de tuples (category_id, display_name).

    Returns:
        category_id sélectionné, ou None si la liste est vide.
    """
    if not categories:
        return None

    options = [cat_id for cat_id, _ in categories]
    labels  = {cat_id: display for cat_id, display in categories}

    # Initialise ou corrige la valeur en session (catégorie supprimée entre deux sessions)
    current = st.session_state.get("selected_category")
    if current not in options:
        st.session_state["selected_category"] = options[0]

    return st.selectbox(
        "Catégorie",
        options=options,
        key="selected_category",
        format_func=lambda c: f"{labels.get(c, c)}  ({c.replace('_', '.')})",
        label_visibility="collapsed",
    )
