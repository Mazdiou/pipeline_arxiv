"""
dashboard/app.py

Point d'entrée du dashboard ArXiv Pipeline.

Usage :
    streamlit run dashboard/app.py

Navigation :
    Page 0 — Contrôle      : gestion des catégories, triggers Airflow, statut live
    Page 1 — Vue catégorie  : KPIs, volume mensuel, top topics, états
    Page 2 — Tendances      : radar par état, cards de scores, tableau comparatif
    Page 3 — Topic          : évolution, 13 métriques, dérives sémantique/lexicale
    Page 4 — Articles       : table filtrée paginée, détail avec NER et mots-clés
"""

import sys
import os

# Garantit que le répertoire racine du projet est dans le path Python,
# quel que soit le répertoire depuis lequel `streamlit run` est lancé.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Charge .env depuis la racine du projet (STORAGE_BACKEND, MINIO_*, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv absent → variables déjà injectées dans l'environnement

import streamlit as st
from dashboard.views import (
    page_control,
    page_category,
    page_trends,
    page_topic,
    page_articles,
)

# ── Configuration globale Streamlit ─────────────────────────────────────────
st.set_page_config(
    page_title="ArXiv Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ───────────────────────────────────────────────────────────────
PAGES = {
    "⚙️  Contrôle":         page_control,
    "📊  Vue catégorie":     page_category,
    "📈  Tendances":         page_trends,
    "🔍  Analyse topic":     page_topic,
    "📄  Articles":          page_articles,
}

with st.sidebar:
    st.markdown("## 🔬 ArXiv Pipeline")
    st.divider()
    selection = st.radio(
        "Navigation",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("v0.1 · dashboard.duckdb")

# ── Rendu de la page sélectionnée ────────────────────────────────────────────
try:
    PAGES[selection].render()
except Exception as exc:
    st.error(
        f"Une erreur est survenue lors du rendu de la page **{selection}** :\n\n"
        f"```\n{exc}\n```\n\n"
        "Vérifiez les logs serveur pour plus de détails.",
        icon="🚨",
    )
    raise
