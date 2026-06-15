from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.core.constants import STATUS_COLORS, STATUS_LABELS, STATUS_ORDER
from dashboard.core.db import (
    get_available_trend_months,
    get_categories_with_data,
    get_trends_overview,
)
from dashboard.core.ui_helpers import render_category_selector


def render() -> None:
    st.title("Radar des tendances")

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

    available_months = get_available_trend_months(category_id)
    if not available_months:
        st.warning(
            f"Aucun rapport de tendance disponible pour **{category_id}**. "
            "Le bootstrap est peut-être encore en cours, ou le pipeline NLP "
            "n'a pas encore produit de résultats.",
            icon="⚠️",
        )
        return

    selected_month = available_months[0]
    if len(available_months) > 1:
        selected_month = st.selectbox(
            "Mois du rapport",
            options=available_months,
            index=0,
            key="trends_month",
            help="Sélectionnez le mois du rapport de tendance à afficher.",
        )

    topics = get_trends_overview(category_id, month=selected_month)

    if not topics:
        st.warning(
            f"Aucun topic trouvé pour le mois **{selected_month}**.",
            icon="⚠️",
        )
        return

    _render_trend_kpis(topics)
    st.divider()
    _render_status_tabs(topics)
    st.divider()
    _render_score_table(topics)


def _render_trend_kpis(topics: list[dict]) -> None:
    counts: dict[str, int] = {}
    for t in topics:
        s = t["status"] or "STABLE"
        counts[s] = counts.get(s, 0) + 1

    present = [(s, counts[s]) for s in STATUS_ORDER if counts.get(s, 0) > 0]
    if not present:
        return

    cols = st.columns(len(present) + 2)

    for col, (status, n) in zip(cols, present):
        color = STATUS_COLORS.get(status, "#6B7280")
        label = STATUS_LABELS.get(status, status)
        col.markdown(
            f"<div style='background:{color}18; border:1px solid {color}40; "
            f"border-radius:8px; padding:8px 10px; text-align:center;'>"
            f"<div style='color:{color}; font-weight:700; font-size:0.75em; "
            f"text-transform:uppercase; letter-spacing:0.05em;'>{label}</div>"
            f"<div style='font-size:1.5em; font-weight:700; margin-top:2px;'>{n}</div>"
            f"<div style='font-size:0.7em; color:#888;'>topic{'s' if n > 1 else ''}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    scores = [t["composite_score"] for t in topics if t["composite_score"] is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        cols[len(present)].metric(
            "Score moyen",
            f"{avg_score:.3f}",
            help="Moyenne du score composite sur tous les topics du mois sélectionné.",
        )

    n_strong = sum(1 for t in topics if (t["composite_score"] or 0) > 0.5)
    cols[len(present) + 1].metric(
        "Signaux forts",
        n_strong,
        help="Topics avec un score composite > 0.5.",
    )


def _render_status_tabs(topics: list[dict]) -> None:
    st.subheader("Topics par état")

    groups: dict[str, list[dict]] = {}
    for t in topics:
        s = t["status"] or "STABLE"
        groups.setdefault(s, []).append(t)

    present_statuses = [s for s in STATUS_ORDER if s in groups]
    if not present_statuses:
        st.caption("Aucun topic avec un état de tendance.")
        return

    tab_labels = [
        f"{STATUS_LABELS.get(s, s)} ({len(groups[s])})" for s in present_statuses
    ]
    tabs = st.tabs(tab_labels)

    for tab, status in zip(tabs, present_statuses):
        with tab:
            color = STATUS_COLORS.get(status, "#6B7280")
            group_topics = sorted(
                groups[status],
                key=lambda t: (
                    t["composite_score"] if t["composite_score"] is not None else -1
                ),
                reverse=True,
            )
            for i in range(0, len(group_topics), 2):
                row_cols = st.columns(2)
                for j, col in enumerate(row_cols):
                    if i + j < len(group_topics):
                        with col:
                            _render_topic_card(group_topics[i + j], color)


def _render_topic_card(topic: dict, color: str) -> None:
    label = topic["topic_label"]
    if len(label) > 80:
        label = label[:77] + "…"

    score = topic["composite_score"]
    score_str = f"{score:.3f}" if score is not None else "—"
    score_pct = int((score or 0) * 100)

    bar_html = (
        f"<div style='background:#e5e7eb; border-radius:4px; height:6px; margin:6px 0;'>"
        f"<div style='background:{color}; width:{min(score_pct, 100)}%; "
        f"height:6px; border-radius:4px;'></div>"
        f"</div>"
    )

    metrics_parts = []
    if topic.get("trend_score") is not None:
        metrics_parts.append(f"Tendance <b>{topic['trend_score']:.3f}</b>")
    if topic.get("weak_signal_score") is not None:
        metrics_parts.append(f"Signal faible <b>{topic['weak_signal_score']:.3f}</b>")
    if topic.get("semantic_score") is not None:
        metrics_parts.append(f"Sémantique <b>{topic['semantic_score']:.3f}</b>")
    metrics_str = " · ".join(metrics_parts) if metrics_parts else ""

    footer = (
        f"{topic['n_articles']:,} articles".replace(",", "\u202f")
        + f" · {topic['n_months_active']} mois actif"
    )

    st.markdown(
        f"<div style='border:1px solid {color}33; border-radius:10px; "
        f"padding:12px 14px; margin-bottom:10px; background:{color}08;'>"
        f"<div style='font-weight:600; font-size:0.9em; line-height:1.3; "
        f"margin-bottom:4px;'>{label}</div>"
        f"<div style='display:flex; align-items:center; gap:10px;'>"
        f"<span style='font-size:1.4em; font-weight:700; color:{color};'>{score_str}</span>"
        f"<div style='flex:1;'>{bar_html}</div>"
        f"</div>"
        f"<div style='font-size:0.75em; color:#888; margin-top:4px;'>{metrics_str}</div>"
        f"<div style='font-size:0.72em; color:#aaa; margin-top:6px;'>{footer}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_score_table(topics: list[dict]) -> None:
    st.subheader("Tableau des scores")

    rows = []
    for t in topics:
        status = t["status"] or "—"
        rows.append(
            {
                "Topic": t["topic_label"],
                "État": STATUS_LABELS.get(status, status),
                "Score composite": t["composite_score"],
                "Tendance": t["trend_score"],
                "Signal faible": t["weak_signal_score"],
                "Sémantique": t["semantic_score"],
                "CAGR": t["cagr"],
                "OLS slope": t["ols_slope"],
                "Persistence": t["persistence_score"],
                "Dérive séma.": t["semantic_drift_last"],
                "Nouveauté moy.": t.get("mean_novelty"),
                "Prob. topic moy.": t.get("mean_topic_prob"),
                "% anglais": t.get("pct_english"),
                "Dist. centroïde": t.get("mean_centroid_distance"),
                "Lisibilité moy.": t.get("mean_flesch_grade"),
                "Densité LaTeX moy.": t.get("mean_latex_density"),
                "% cross-cat.": t.get("pct_cross_category"),
                "Articles": t["n_articles"],
                "Mois actif": t["n_months_active"],
            }
        )

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Topic": st.column_config.TextColumn("Topic", width="large"),
            "État": st.column_config.TextColumn("État", width="medium"),
            "Score composite": st.column_config.NumberColumn(
                "Score composite", format="%.4f", width="small"
            ),
            "Tendance": st.column_config.NumberColumn(
                "Tendance", format="%.4f", width="small"
            ),
            "Signal faible": st.column_config.NumberColumn(
                "Signal faible", format="%.4f", width="small"
            ),
            "Sémantique": st.column_config.NumberColumn(
                "Sémantique", format="%.4f", width="small"
            ),
            "CAGR": st.column_config.NumberColumn("CAGR", format="%.4f", width="small"),
            "OLS slope": st.column_config.NumberColumn(
                "OLS slope", format="%.4f", width="small"
            ),
            "Persistence": st.column_config.NumberColumn(
                "Persistence", format="%.4f", width="small"
            ),
            "Dérive séma.": st.column_config.NumberColumn(
                "Dérive séma.", format="%.4f", width="small"
            ),
            "Nouveauté moy.": st.column_config.NumberColumn(
                "Nouveauté moy.", format="%.4f", width="small"
            ),
            "Prob. topic moy.": st.column_config.NumberColumn(
                "Prob. topic moy.", format="%.4f", width="small"
            ),
            "% anglais": st.column_config.NumberColumn(
                "% anglais", format="%.1f", width="small"
            ),
            "Dist. centroïde": st.column_config.NumberColumn(
                "Dist. centroïde", format="%.4f", width="small"
            ),
            "Lisibilité moy.": st.column_config.NumberColumn(
                "Lisibilité moy.", format="%.2f", width="small"
            ),
            "Densité LaTeX moy.": st.column_config.NumberColumn(
                "Densité LaTeX moy.", format="%.4f", width="small"
            ),
            "% cross-cat.": st.column_config.NumberColumn(
                "% cross-cat.", format="%.1f", width="small"
            ),
            "Articles": st.column_config.NumberColumn("Articles", width="small"),
            "Mois actif": st.column_config.NumberColumn("Mois actif", width="small"),
        },
        height=min(600, max(200, len(df) * 35 + 40)),
    )

    st.caption(
        f"{len(df)} topics · mois sélectionné · "
        "Triés par score composite décroissant"
    )
