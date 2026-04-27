"""
dashboard/views/page_category.py

Page 1 — Vue catégorie.

Affiche une vue d'ensemble d'une catégorie arXiv :
    - Sélecteur de catégorie (partagé avec les autres pages via session_state)
    - KPI bar : articles, topics, mois, nouveauté moyenne, % anglais
    - Volume mensuel : bar chart d'articles par mois
    - Top topics : bar chart horizontal coloré par état
    - États des topics : badges EMERGING / TRENDING / STABLE / …

Architecture interne :
    render()
    _render_kpis()
    _render_volume_chart()
    _render_topics_chart()
    _render_status_badges()
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.core.constants import STATUS_COLORS, STATUS_LABELS
from dashboard.core.db import (
    get_categories_with_data,
    get_category_overview,
    get_monthly_volumes,
    get_status_counts,
    get_top_topics,
)
from dashboard.core.ui_helpers import render_category_selector, truncate_label


# =========================================================================== #
# Point d'entrée public                                                        #
# =========================================================================== #

def render() -> None:
    """Point d'entrée de la Page 1 — appelé par app.py."""
    st.title("Vue catégorie")

    categories = get_categories_with_data()

    if not categories:
        st.info(
            "Aucune catégorie disponible. "
            "Configurez et bootstrappez une catégorie depuis la page **Contrôle**.",
            icon="ℹ️",
        )
        return

    category_id = render_category_selector(categories)
    if not category_id:
        return

    overview = get_category_overview(category_id)
    if not overview or overview.get("n_articles", 0) == 0:
        st.warning(
            f"Aucun article trouvé pour **{category_id}**. "
            "Le bootstrap est peut-être encore en cours.",
            icon="⚠️",
        )
        return

    _render_kpis(overview)
    st.divider()
    _render_volume_chart(category_id)
    st.divider()
    _render_topics_chart(category_id)
    st.divider()
    _render_status_badges(category_id)


# =========================================================================== #
# KPI bar                                                                      #
# =========================================================================== #

def _render_kpis(overview: dict) -> None:
    """Affiche les 5 métriques clés de la catégorie."""
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Articles",
        f"{overview['n_articles']:,}".replace(",", " "),
    )
    c2.metric(
        "Topics",
        overview["n_topics"],
    )
    c3.metric(
        "Mois couverts",
        overview["n_months"],
        help=(
            f"De {overview['date_min']} à {overview['date_max']}"
            if overview.get("date_min")
            else None
        ),
    )
    nov = overview.get("avg_novelty")
    c4.metric(
        "Nouveauté moy.",
        f"{nov:.3f}" if nov is not None else "—",
        help="Score de nouveauté moyen des articles (0 = très similaire aux existants, 1 = très nouveau).",
    )
    pct = overview.get("pct_english")
    c5.metric(
        "Anglais",
        f"{pct:.0f} %" if pct is not None else "—",
    )


# =========================================================================== #
# Volume mensuel                                                               #
# =========================================================================== #

def _render_volume_chart(category: str) -> None:
    """Bar chart du nombre d'articles publiés par mois."""
    st.subheader("Volume mensuel")

    data = get_monthly_volumes(category)
    if not data:
        st.caption("Pas de données de volume disponibles.")
        return

    months     = [d["month"] for d in data]
    n_articles = [d["n_articles"] for d in data]

    fig = go.Figure(
        go.Bar(
            x=months,
            y=n_articles,
            marker_color="#3B82F6",
            hovertemplate="<b>%{x}</b><br>%{y} articles<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        xaxis_title=None,
        yaxis_title="Articles",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================================== #
# Top topics                                                                   #
# =========================================================================== #

def _render_topics_chart(category: str) -> None:
    """Bar chart horizontal des topics les plus actifs."""
    st.subheader("Top topics")

    topics = get_top_topics(category, limit=20)
    if not topics:
        st.caption("Aucun topic disponible.")
        return

    # Tronque les labels trop longs pour l'axe Y
    labels  = [truncate_label(t["topic_label"]) for t in topics]
    values  = [t["n_articles"] for t in topics]
    colors  = [
        STATUS_COLORS.get(t["last_status"] or "", "#6B7280")
        for t in topics
    ]
    hovers  = [
        f"<b>{t['topic_label']}</b><br>"
        f"{t['n_articles']:,} articles · {t['n_months_active']} mois actif"
        f"{'<br>État : ' + STATUS_LABELS.get(t['last_status'], t['last_status']) if t['last_status'] else ''}"
        "<extra></extra>"
        for t in topics
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate=hovers,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=max(300, len(topics) * 28),
        xaxis_title="Articles",
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Légende — seulement les états présents dans les données
    present = {t["last_status"] for t in topics if t["last_status"]}
    if present:
        st.caption(
            "  ".join(
                f"<span style='color:{STATUS_COLORS[s]}'>■</span> {STATUS_LABELS.get(s, s)}"
                for s in STATUS_COLORS
                if s in present
            ),
            unsafe_allow_html=True,
        )


# =========================================================================== #
# États des topics                                                             #
# =========================================================================== #

def _render_status_badges(category: str) -> None:
    """
    Affiche les compteurs de topics par état sous forme de badges en colonnes.
    Seuls les états présents dans les données sont affichés.
    """
    st.subheader("États des topics")

    counts = get_status_counts(category)
    if not counts:
        st.caption("Aucun rapport de tendances disponible.")
        return

    # Construire la liste des états présents dans l'ordre voulu
    order   = ["EMERGING", "TRENDING", "TRANSFORMING", "STABLE", "DECLINING", "NEW"]
    present = [(s, counts[s]) for s in order if counts.get(s, 0) > 0]

    if not present:
        st.caption("Aucun état de tendance enregistré.")
        return

    cols = st.columns(len(present))
    for col, (status, n) in zip(cols, present):
        color = STATUS_COLORS.get(status, "#6B7280")
        label = STATUS_LABELS.get(status, status)
        col.markdown(
            f"<div style='background:{color}18; border:1px solid {color}40; "
            f"border-radius:10px; padding:10px 14px; text-align:center;'>"
            f"<div style='color:{color}; font-weight:700; font-size:0.85em; "
            f"text-transform:uppercase; letter-spacing:0.05em;'>{label}</div>"
            f"<div style='font-size:1.6em; font-weight:700; margin-top:4px;'>{n}</div>"
            f"<div style='font-size:0.75em; color:#888;'>topic{'s' if n > 1 else ''}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
