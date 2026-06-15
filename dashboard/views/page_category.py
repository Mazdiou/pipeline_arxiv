from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.core.constants import (
    STATUS_COLORS,
    STATUS_DESCRIPTIONS,
    STATUS_LABELS,
    fmt_number,
)
from dashboard.core.db import (
    get_categories_with_data,
    get_category_overview,
    get_monthly_volumes,
    get_status_counts,
    get_top_topics,
)
from dashboard.core.ui_helpers import render_category_selector, truncate_label


def render() -> None:
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


def _render_kpis(overview: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Articles",
        fmt_number(overview["n_articles"]),
    )
    c2.metric(
        "Topics",
        overview["n_topics"],
        help="Nombre de topics BERTopic détectés sur l'ensemble des données de cette catégorie.",
    )
    months_help = None
    if overview.get("date_min") and overview.get("date_max"):
        months_help = (
            f"Plage : {overview['date_min']} → {overview['date_max']}. "
            "Attention : peut contenir des mois sans données si la collecte était inactive."
        )
    c3.metric(
        "Mois avec données",
        overview["n_months"],
        help=months_help,
    )
    nov = overview.get("avg_novelty")
    c4.metric(
        "Nouveauté moy.",
        f"{nov:.3f}" if nov is not None else "—",
        help=(
            "Score d'isolement sémantique moyen des articles (0 → 1). "
            "0 = très similaire aux articles existants du topic, "
            "1 = très différent (potentiellement nouveau angle de recherche). "
            "Calculé à partir des embeddings par le module Enricher. "
            "Voir la page **Analyse topic** pour la formule détaillée."
        ),
    )
    pct = overview.get("pct_english")
    c5.metric(
        "% anglais",
        f"{pct:.0f} %" if pct is not None else "—",
        help="Pourcentage d'articles dont la langue détectée est l'anglais.",
    )


def _render_volume_chart(category: str) -> None:
    st.subheader("Volume mensuel")

    data = get_monthly_volumes(category)
    if not data:
        st.caption("Pas de données de volume disponibles.")
        return

    months = [d["month"] for d in data]
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


def _render_topics_chart(category: str) -> None:
    st.subheader("Top topics")

    topics = get_top_topics(category, limit=20)
    if not topics:
        st.caption("Aucun topic disponible.")
        return

    labels = [truncate_label(t["topic_label"]) for t in topics]
    values = [t["n_articles"] for t in topics]
    colors = [STATUS_COLORS.get(t["last_status"] or "", "#6B7280") for t in topics]
    hovers = []
    for t in topics:
        n_art = f"{t['n_articles']:,}".replace(",", "\u202f")
        status_line = ""
        if t.get("last_status"):
            status_label = STATUS_LABELS.get(t["last_status"], t["last_status"])
            status_desc = STATUS_DESCRIPTIONS.get(t["last_status"], "")
            status_line = f"<br>État : {status_label}"
            if status_desc:
                status_line += f"<br><i>{status_desc}</i>"
        hovers.append(
            f"<b>{t['topic_label']}</b><br>"
            f"{n_art} articles · {t['n_months_active']} mois actif"
            f"{status_line}"
            "<extra></extra>"
        )

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


def _render_status_badges(category: str) -> None:
    st.subheader("États des topics")

    counts = get_status_counts(category)
    if not counts:
        st.caption("Aucun rapport de tendances disponible.")
        return

    order = ["EMERGING", "TRENDING", "TRANSFORMING", "STABLE", "DECLINING", "NEW"]
    present = [(s, counts[s]) for s in order if counts.get(s, 0) > 0]

    if not present:
        st.caption("Aucun état de tendance enregistré.")
        return

    cols = st.columns(len(present))
    for col, (status, n) in zip(cols, present):
        color = STATUS_COLORS.get(status, "#6B7280")
        label = STATUS_LABELS.get(status, status)
        desc = STATUS_DESCRIPTIONS.get(status, "")
        topic_str = "topic" if n == 1 else "topics"
        col.markdown(
            f"<div style='background:{color}18; border:1px solid {color}40; "
            f"border-radius:10px; padding:10px 14px; text-align:center;' "
            f"title='{desc}'>"
            f"<div style='color:{color}; font-weight:700; font-size:0.85em; "
            f"text-transform:uppercase; letter-spacing:0.05em;'>{label}</div>"
            f"<div style='font-size:1.6em; font-weight:700; margin-top:4px;'>{n}</div>"
            f"<div style='font-size:0.75em; color:#888;'>{topic_str}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
