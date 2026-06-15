from __future__ import annotations

import html as _html_module
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.constants import STATUS_COLORS, STATUS_LABELS
from dashboard.core.db import (
    get_categories_with_data,
    get_topic_detail,
    get_topic_monthly_stats,
    get_topics_for_category,
    get_trend_full_history,
)
from dashboard.core.ui_helpers import render_category_selector

_STATUS_ICONS: dict[str, str] = {
    "NEW": "🆕",
    "EMERGING": "🌱",
    "TRENDING": "📈",
    "TRANSFORMING": "🔄",
    "STABLE": "⚡",
    "DECLINING": "📉",
}

_METRIC_INFO: dict[str, tuple[str, str]] = {
    "composite_score": (
        "Score composite",
        "Score global de tendance (0–1). Combine les sous-scores pondérés.",
    ),
    "trend_score": (
        "Score tendance",
        "Intensité et régularité de la croissance du volume.",
    ),
    "weak_signal_score": (
        "Signal faible",
        "Détecte les topics en émergence précoce (faible volume mais en hausse).",
    ),
    "semantic_score": (
        "Score sémantique",
        "Évolution du contenu sémantique des articles au fil du temps.",
    ),
    "z_score_last": (
        "Z-score",
        "Écart-type de l'activité récente par rapport à la moyenne historique.",
    ),
    "ols_slope": (
        "Pente OLS",
        "Tendance linéaire par régression (articles/mois). Positif = croissance.",
    ),
    "mann_kendall_tau": (
        "Mann-Kendall τ",
        "Corrélation de rang sur la série temporelle. +1 = tendance monotone croissante.",
    ),
    "mann_kendall_pvalue": (
        "Mann-Kendall p-value",
        "Significativité statistique du test de tendance (< 0.05 = significatif).",
    ),
    "hurst_exponent": (
        "Exposant de Hurst",
        "Persistance de la série (> 0.5 = tendance persistante, < 0.5 = retour à la moyenne).",
    ),
    "cagr": ("CAGR", "Taux de croissance annuel composé du volume d'articles."),
    "burst_state_last": (
        "État de burst",
        "État du burst de Kleinberg (0=Inactif, 1=Actif, 2=Intense).",
    ),
    "emergence_score_last": (
        "Score d'émergence",
        "Force du signal d'émergence basé sur les pics de Kleinberg.",
    ),
    "novelty_weighted_emergence_last": (
        "Émergence pondérée",
        "Émergence pondérée par la nouveauté moyenne des articles.",
    ),
    "persistence_score": (
        "Persistance",
        "Stabilité de la présence du topic dans le temps (0 = éphémère, 1 = permanent).",
    ),
    "semantic_drift_last": (
        "Dérive sémantique",
        "Distance cosinus entre les embeddings récents et anciens.",
    ),
    "keyword_drift_last": (
        "Dérive lexicale",
        "Taux de renouvellement des mots-clés entre les périodes récente et ancienne.",
    ),
    "stability_score": (
        "Score de stabilité",
        "Inverse de la variance du volume. Élevé = volume régulier.",
    ),
    "mean_novelty": (
        "Nouveauté moy.",
        "Score de nouveauté moyen des articles de ce topic sur le mois du rapport.",
    ),
    "mean_topic_prob": (
        "Prob. topic moy.",
        "Probabilité d'appartenance moyenne au topic BERTopic (cohésion thématique).",
    ),
    "mean_centroid_distance": (
        "Dist. centroïde moy.",
        "Distance euclidienne moyenne au centroïde du topic (dispersion sémantique).",
    ),
    "pct_english": ("% anglais", "Pourcentage d'articles en anglais dans ce topic."),
    "mean_flesch_grade": (
        "Lisibilité moy.",
        "Score Flesch-Kincaid moyen (niveau scolaire). Élevé = texte plus difficile.",
    ),
    "mean_latex_density": (
        "Densité LaTeX moy.",
        "Proportion moyenne de tokens LaTeX dans les abstracts.",
    ),
    "pct_cross_category": (
        "% cross-catégorie",
        "Pourcentage d'articles appartenant aussi à d'autres catégories arXiv.",
    ),
}


def render() -> None:
    pending = st.session_state.pop("_pending_nav", None)
    if pending and "category" in pending and "topic_id" in pending:
        st.session_state["selected_category"] = pending["category"]
        st.session_state["_topic_sel_category"] = pending["category"]
        st.session_state["selected_topic_id"] = pending["topic_id"]

    st.title("Analyse d'un topic")

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

    topics = get_topics_for_category(category_id)
    if not topics:
        st.warning(
            f"Aucun topic disponible pour **{category_id}**. "
            "Le bootstrap est peut-être encore en cours.",
            icon="⚠️",
        )
        return

    topic_id = _render_topic_selector(topics, category_id)
    if topic_id is None:
        return

    detail = get_topic_detail(category_id, topic_id)
    monthly = get_topic_monthly_stats(category_id, topic_id)
    full_history = get_trend_full_history(category_id, topic_id)

    if not detail:
        topic_info = next((t for t in topics if t["topic_id"] == topic_id), {})
        n_months = topic_info.get("n_months_active", 0)
        label = topic_info.get("topic_label", f"Topic {topic_id}")
        st.info(
            f"**{_html_module.escape(label)}** n'a pas encore de rapport de tendance.\n\n"
            f"Ce topic est actif depuis **{n_months} mois**. "
            f"Les métriques de tendance nécessitent un historique minimum "
            f"configuré dans le pipeline (`min_months_for_trend`).\n\n"
            f"Relancez un bootstrap complet ou attendez les prochaines syncs mensuelles.",
            icon="ℹ️",
        )
        if monthly:
            st.divider()
            st.subheader("Volume mensuel disponible")
            _render_evolution_charts(monthly, [])
        return

    _render_topic_header(detail)
    st.divider()
    _render_evolution_charts(monthly, full_history)
    st.divider()
    _render_metrics_grid(detail)
    st.divider()
    _render_drift_chart(full_history)

    if full_history:
        with st.expander("Historique mensuel complet", expanded=False):
            _render_history_table(full_history)


def _render_topic_selector(topics: list[dict], category_id: str) -> Optional[int]:
    if st.session_state.get("_topic_sel_category") != category_id:
        st.session_state["_topic_sel_category"] = category_id
        st.session_state.pop("selected_topic_id", None)

    topic_map = {t["topic_id"]: t for t in topics}
    options = [t["topic_id"] for t in topics]

    current = st.session_state.get("selected_topic_id")
    if options and current not in options:
        st.session_state["selected_topic_id"] = options[0]

    def _fmt(tid: int) -> str:
        t = topic_map.get(tid, {})
        status = t.get("status") or ""
        icon = _STATUS_ICONS.get(status, "○")
        label = t.get("topic_label", f"Topic {tid}")
        score = t.get("composite_score")
        score_str = f"  [{score:.3f}]" if score is not None else "  [pas de rapport]"
        return f"{icon}  {label}{score_str}"

    return st.selectbox(
        "Topic",
        options=options,
        key="selected_topic_id",
        format_func=_fmt,
        label_visibility="collapsed",
        help="Topics triés par score composite décroissant. [pas de rapport] = topic sans métriques de tendance.",
    )


def _render_topic_header(detail: dict) -> None:
    status = detail.get("status") or "STABLE"
    color = STATUS_COLORS.get(status, "#6B7280")
    label = STATUS_LABELS.get(status, status)
    icon = _STATUS_ICONS.get(status, "○")

    safe_label = _html_module.escape(detail.get("topic_label", "—"))

    st.markdown(
        f"### {safe_label}\n"
        f"<span style='background:{color}22; color:{color}; border:1px solid {color}55; "
        f"border-radius:6px; padding:3px 10px; font-size:0.85em; font-weight:600;'>"
        f"{icon} {label}</span>",
        unsafe_allow_html=True,
    )
    n_months = detail.get("n_months_active", 0)
    n_articles = detail.get("n_articles", 0)
    st.caption(
        f"Période : {detail.get('first_month', '—')} → {detail.get('last_month', '—')}  "
        f"·  {n_months} mois actif  "
        f"·  {n_articles:,} articles  ".replace(",", "\u202f")
        + f"·  Rapport du {detail.get('report_month', '—')}"
    )

    c1, c2, c3, c4 = st.columns(4)
    _metric_col(c1, "composite_score", detail)
    _metric_col(c2, "trend_score", detail)
    _metric_col(c3, "weak_signal_score", detail)
    _metric_col(c4, "semantic_score", detail)


def _metric_col(col, key: str, detail: dict) -> None:
    name, help_text = _METRIC_INFO.get(key, (key, ""))
    val = detail.get(key)
    col.metric(
        label=name,
        value=f"{val:.4f}" if val is not None else "—",
        help=help_text,
    )


def _render_evolution_charts(monthly: list[dict], full_history: list[dict]) -> None:
    st.subheader("Évolution temporelle")

    col_vol, col_score = st.columns(2)

    with col_vol:
        st.caption("**Volume mensuel** (articles)")
        if not monthly:
            st.caption("Pas de données de volume.")
        else:
            months = [d["month"] for d in monthly]
            n_articles = [d["n_articles"] for d in monthly]
            novelty = [d["mean_novelty"] for d in monthly]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=months,
                    y=n_articles,
                    name="Articles",
                    marker_color="#3B82F6",
                    hovertemplate="<b>%{x}</b><br>%{y} articles<extra></extra>",
                )
            )
            if any(v is not None for v in novelty):
                fig.add_trace(
                    go.Scatter(
                        x=months,
                        y=novelty,
                        name="Nouveauté moy.",
                        mode="lines+markers",
                        line=dict(color="#F59E0B", width=2),
                        yaxis="y2",
                        hovertemplate="<b>%{x}</b><br>Nouveauté : %{y:.3f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    yaxis2=dict(
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        range=[0, 1],
                        tickfont=dict(color="#F59E0B"),
                    ),
                )
            fig.update_layout(**_chart_layout(height=260))
            st.plotly_chart(fig, use_container_width=True)

    with col_score:
        st.caption("**Scores de tendance** dans le temps")
        if not full_history:
            st.caption("Pas d'historique de tendance.")
        else:
            months = [d["month"] for d in full_history]
            fig = go.Figure()
            traces = [
                ("composite_score", "Score composite", "#3B82F6", 3),
                ("trend_score", "Tendance", "#00C49A", 2),
                ("weak_signal_score", "Signal faible", "#8B5CF6", 2),
                ("semantic_score", "Sémantique", "#F59E0B", 2),
            ]
            for key, name, color, width in traces:
                vals = [d[key] for d in full_history]
                if any(v is not None for v in vals):
                    fig.add_trace(
                        go.Scatter(
                            x=months,
                            y=vals,
                            name=name,
                            mode="lines+markers",
                            line=dict(color=color, width=width),
                            hovertemplate=f"<b>%{{x}}</b><br>{name} : %{{y:.4f}}<extra></extra>",
                        )
                    )
            fig.update_layout(
                **_chart_layout(height=260),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.01,
                    xanchor="left",
                    x=0,
                    font=dict(size=11),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_metrics_grid(detail: dict) -> None:
    st.subheader("Métriques détaillées")
    st.caption(f"Rapport du mois : **{detail.get('report_month', '—')}**")

    trend_metrics = [
        "z_score_last",
        "mann_kendall_tau",
        "mann_kendall_pvalue",
        "hurst_exponent",
        "cagr",
        "burst_state_last",
        "emergence_score_last",
        "semantic_drift_last",
        "keyword_drift_last",
        "stability_score",
    ]

    cols = st.columns(4)
    for i, key in enumerate(trend_metrics):
        col = cols[i % 4]
        name, help_text = _METRIC_INFO.get(key, (key, ""))
        val = detail.get(key)

        if key == "burst_state_last":
            burst_labels = {0: "Inactif", 1: "Actif", 2: "Intense"}
            display = burst_labels.get(val, str(val)) if val is not None else "—"
        elif key == "mann_kendall_pvalue":
            display = f"{val:.4f}" if val is not None else "—"
        elif val is not None:
            display = f"{val:.4f}"
        else:
            display = "—"

        col.metric(label=name, value=display, help=help_text)

    enricher_metrics = [
        ("mean_novelty", "%.4f"),
        ("mean_topic_prob", "%.4f"),
        ("mean_centroid_distance", "%.4f"),
        ("pct_english", "%.1f %%"),
        ("mean_flesch_grade", "%.2f"),
        ("mean_latex_density", "%.4f"),
        ("pct_cross_category", "%.1f %%"),
    ]

    has_enricher = any(detail.get(k) is not None for k, _ in enricher_metrics)
    if has_enricher:
        st.markdown(
            "<div style='margin-top:16px; margin-bottom:4px; font-size:0.85em; "
            "font-weight:600; color:#888; text-transform:uppercase; "
            "letter-spacing:0.06em;'>Caractéristiques NLP des articles</div>",
            unsafe_allow_html=True,
        )
        ecols = st.columns(4)
        for i, (key, fmt) in enumerate(enricher_metrics):
            col = ecols[i % 4]
            name, help_text = _METRIC_INFO.get(key, (key, ""))
            val = detail.get(key)
            try:
                display = (fmt % val) if val is not None else "—"
            except (TypeError, ValueError):
                display = "—"
            col.metric(label=name, value=display, help=help_text)


def _render_drift_chart(full_history: list[dict]) -> None:
    if not full_history:
        return

    has_drift = any(
        d["semantic_drift_last"] is not None or d["keyword_drift_last"] is not None
        for d in full_history
    )
    if not has_drift:
        return

    st.subheader("Dérives sémantique et lexicale")

    months = [d["month"] for d in full_history]
    sem_drift = [d["semantic_drift_last"] for d in full_history]
    kw_drift = [d["keyword_drift_last"] for d in full_history]
    stability = [d["stability_score"] for d in full_history]

    fig = go.Figure()

    if any(v is not None for v in sem_drift):
        fig.add_trace(
            go.Scatter(
                x=months,
                y=sem_drift,
                name="Dérive sémantique",
                mode="lines+markers",
                line=dict(color="#8B5CF6", width=2),
                hovertemplate="<b>%{x}</b><br>Dérive séma. : %{y:.4f}<extra></extra>",
            )
        )
    if any(v is not None for v in kw_drift):
        fig.add_trace(
            go.Scatter(
                x=months,
                y=kw_drift,
                name="Dérive lexicale",
                mode="lines+markers",
                line=dict(color="#EF4444", width=2),
                hovertemplate="<b>%{x}</b><br>Dérive lex. : %{y:.4f}<extra></extra>",
            )
        )
    if any(v is not None for v in stability):
        fig.add_trace(
            go.Scatter(
                x=months,
                y=stability,
                name="Stabilité",
                mode="lines",
                line=dict(color="#6B7280", width=1, dash="dot"),
                hovertemplate="<b>%{x}</b><br>Stabilité : %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        **_chart_layout(height=260),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "La **dérive sémantique** mesure l'évolution du sens (embeddings). "
        "La **dérive lexicale** mesure le renouvellement des mots-clés. "
        "Une valeur élevée indique un topic en transformation profonde."
    )


def _render_history_table(full_history: list[dict]) -> None:
    rows = []
    for d in full_history:
        status = d["status"] or "—"
        rows.append(
            {
                "Mois": d["month"],
                "État": STATUS_LABELS.get(status, status),
                "Composite": d["composite_score"],
                "Tendance": d["trend_score"],
                "Signal faible": d["weak_signal_score"],
                "Sémantique": d["semantic_score"],
                "Z-score": d["z_score_last"],
                "OLS slope": d["ols_slope"],
                "CAGR": d["cagr"],
                "Émergence": d["emergence_score_last"],
                "Persistence": d["persistence_score"],
                "Dérive séma.": d["semantic_drift_last"],
                "Dérive lex.": d["keyword_drift_last"],
                "Stabilité": d["stability_score"],
            }
        )

    df = pd.DataFrame(rows)
    float_cols = [c for c in df.columns if c not in ("Mois", "État")]
    col_config = {
        "Mois": st.column_config.TextColumn("Mois", width="small"),
        "État": st.column_config.TextColumn("État", width="medium"),
    }
    col_config.update(
        {
            c: st.column_config.NumberColumn(c, format="%.4f", width="small")
            for c in float_cols
        }
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        height=min(500, max(200, len(df) * 35 + 40)),
    )


def _chart_layout(height: int = 280) -> dict:
    return dict(
        margin=dict(l=0, r=0, t=10, b=0),
        height=height,
        xaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
