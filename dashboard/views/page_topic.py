"""
dashboard/views/page_topic.py

Page 3 — Analyse d'un topic.

Affiche l'analyse complète d'un topic arXiv sur sa durée de vie :
    - Sélecteur catégorie + sélecteur topic (trié par score composite)
    - Header : état actuel, score composite, période couverte
    - Évolution : volume mensuel + scores composites dans le temps
    - 13 métriques détaillées du dernier rapport
    - Dérives sémantique et lexicale dans le temps
    - Historique mensuel complet (tableau dépliable)

Architecture interne :
    render()
    _render_topic_selector()
    _render_topic_header()
    _render_evolution_charts()
    _render_metrics_grid()
    _render_drift_chart()
    _render_history_table()
"""

from __future__ import annotations

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

# Icônes par état pour le sélecteur de topics
_STATUS_ICONS: dict[str, str] = {
    "NEW":          "🆕",
    "EMERGING":     "🌱",
    "TRENDING":     "📈",
    "TRANSFORMING": "🔄",
    "STABLE":       "⚡",
    "DECLINING":    "📉",
}

# Descriptions courtes des métriques (tooltip / label)
_METRIC_INFO: dict[str, tuple[str, str]] = {
    "composite_score": (
        "Score composite",
        "Score global de tendance ∈ [0, 1]. Moyenne pondérée des trois sous-scores : "
        "Tendance (volume), Signal faible (émergence), Sémantique (évolution du contenu). "
        "< 0.3 : faible activité · 0.3–0.6 : modéré · > 0.6 : signal fort.",
    ),
    "trend_score": (
        "Score tendance",
        "Intensité et régularité de la croissance du volume d'articles. "
        "Combine la pente OLS normalisée, le test Mann-Kendall et l'exposant de Hurst. "
        "∈ [0, 1] : 0 = aucune croissance, 1 = croissance forte et persistante.",
    ),
    "weak_signal_score": (
        "Signal faible",
        "Détecte les topics en émergence précoce : fort taux de croissance malgré un faible volume absolu. "
        "Combine le score d'émergence (croissance / log(volume)) et le burst de Kleinberg. "
        "∈ [0, 1] : élevé = topic naissant en accélération, candidat à surveiller.",
    ),
    "semantic_score": (
        "Score sémantique",
        "Mesure l'évolution du contenu des articles dans le temps via leurs embeddings. "
        "Une dérive sémantique croissante signale un topic qui se transforme ou qui absorbe "
        "de nouveaux sous-domaines. ∈ [0, 1] : > 0.5 = contenu en mutation active.",
    ),
    "z_score_last": (
        "Z-score (glissant)",
        "Mesure à quel point le volume du dernier mois est exceptionnel par rapport "
        "aux 6 mois précédents.\n"
        "Formule : z = (volume_actuel − μ) / σ  (μ et σ calculés sur la fenêtre glissante)\n"
        "Interprétation :\n"
        "  z > +2 → pic anormal (top 2.5 % de la fenêtre)\n"
        "  z > +1 → au-dessus de la normale\n"
        "  |z| ≤ 1 → dans la plage habituelle\n"
        "  z < −2 → creux anormal (baisse inhabituelle)",
    ),
    "ols_slope": (
        "Pente OLS (articles/mois)",
        "Pente de la droite de régression linéaire (Ordinary Least Squares) ajustée sur toute la série.\n"
        "Formule : y = a·t + b — la pente a est rapportée ici.\n"
        "Interprétation :\n"
        "  > 0 → croissance linéaire (ex : +3.5 = 3,5 articles de plus par mois)\n"
        "  ≈ 0 → volume stable\n"
        "  < 0 → déclin\n"
        "Limite : ne capte pas les tendances exponentielles ou en plateau.",
    ),
    "mann_kendall_tau": (
        "Mann-Kendall τ",
        "Test non-paramétrique de tendance monotone sur la série temporelle entière.\n"
        "Principe : pour chaque paire (i,j), vote +1 si la série monte, −1 si elle descend. "
        "τ est la proportion normalisée de ces votes.\n"
        "Interprétation :\n"
        "  τ ∈ [−1, +1]\n"
        "  +1 → tendance parfaitement croissante\n"
        "  −1 → tendance parfaitement décroissante\n"
        "  0  → aucune tendance détectable\n"
        "Avantage vs OLS : robuste aux outliers et aux tendances non-linéaires.",
    ),
    "mann_kendall_pvalue": (
        "Mann-Kendall p-value",
        "Probabilité d'obtenir ce résultat par hasard si la série était aléatoire.\n"
        "Interprétation :\n"
        "  < 0.05 → tendance statistiquement significative (5 % de risque d'erreur)\n"
        "  < 0.01 → très significative\n"
        "  ≥ 0.05 → tendance non confirmée statistiquement\n"
        "Toujours interpréter conjointement avec τ : un τ élevé avec p ≥ 0.05 "
        "peut indiquer une série trop courte.",
    ),
    "hurst_exponent": (
        "Exposant de Hurst",
        "Mesure si la série a une 'mémoire' : une hausse récente annonce-t-elle "
        "de futures hausses, ou un retour à la moyenne ?\n"
        "Méthode R/S (Rescaled Range — Hurst, 1951).\n"
        "Interprétation :\n"
        "  H > 0.6 → série PERSISTANTE : les hausses tendent à continuer\n"
        "  H = 0.5 → marche aléatoire, futur indépendant du passé\n"
        "  H < 0.4 → série ANTI-PERSISTANTE : les hausses alternent avec des baisses\n"
        "Combinaison idéale : Mann-Kendall τ > 0.4 ET Hurst > 0.6 → tendance très fiable.",
    ),
    "cagr": (
        "CAGR (croissance annuelle)",
        "Taux de Croissance Annuel Composé — normalise la croissance sur 12 mois "
        "pour comparer des topics d'âges différents.\n"
        "Formule : CAGR = (V_final / V_initial)^(12/n_mois) − 1\n"
        "Exemples :\n"
        "  0.41 → +41 %/an\n"
        "  2.0  → triplement du volume en un an\n"
        "  −0.3 → −30 %/an (déclin)\n"
        "Limite : ne capture pas la volatilité intra-période.",
    ),
    "burst_state_last": (
        "État de burst (Kleinberg)",
        "Détecte les pics d'activité inhabituels par un automate à états (Kleinberg, 2002). "
        "Modélise la série comme émise par un processus Poisson à taux variable.\n"
        "États :\n"
        "  0 = Inactif  → volume normal\n"
        "  1 = Actif    → burst de niveau 1 (volume ≈ 2× la moyenne historique)\n"
        "  2 = Intense  → burst intense (volume ≈ 4× la moyenne)\n"
        "Un burst actif combiné à un faible volume absolu = signal faible potentiel.",
    ),
    "emergence_score_last": (
        "Score d'émergence",
        "Identifie les topics à forte croissance relative malgré un faible volume.\n"
        "Formule : émergence = taux_de_croissance / log(1 + n_articles)\n"
        "Intuition : un topic qui double en passant de 5 à 10 articles est plus "
        "'émergent' qu'un topic qui gagne 50 articles sur 1000.\n"
        "Élevé → topic naissant en accélération · Négatif → déclin.",
    ),
    "novelty_weighted_emergence_last": (
        "Émergence pondérée (NWE)",
        "Émergence amplifiée par la nouveauté sémantique des articles.\n"
        "Formule : NWE = mean_novelty × taux_de_croissance\n"
        "Deux dimensions :\n"
        "  - mean_novelty : les articles sont-ils sémantiquement isolés des autres clusters ? "
        "(1.0 = très originaux)\n"
        "  - taux_de_croissance : le topic croît-il rapidement ?\n"
        "NWE élevé → topic original ET en plein essor : candidat signal faible prioritaire.",
    ),
    "persistence_score": (
        "Persistance (momentum local)",
        "Proportion de périodes consécutives où la variation garde le même signe.\n"
        "Formule : count(sign(Δt) == sign(Δt-1)) / (n − 2)\n"
        "Interprétation :\n"
        "  1.0 → croissance parfaitement monotone (hausse après hausse)\n"
        "  0.5 → comportement aléatoire (aussi souvent cohérent qu'incohérent)\n"
        "  0.0 → alternances parfaites hausse/baisse (bruit oscillatoire)\n"
        "Différence avec Hurst : Hurst mesure la mémoire longue, Persistence mesure "
        "le momentum à 1 pas de temps.",
    ),
    "semantic_drift_last": (
        "Dérive sémantique",
        "Distance cosinus entre les embeddings des articles récents et anciens du topic.\n"
        "Mesure combien le 'sens' du topic a changé depuis son apparition.\n"
        "Interprétation :\n"
        "  ≈ 0.0 → contenu stable, topic homogène\n"
        "  0.2–0.4 → évolution modérée (absorption de sous-domaines)\n"
        "  > 0.5 → transformation profonde (possible scission ou pivot thématique)",
    ),
    "keyword_drift_last": (
        "Dérive lexicale",
        "Taux de renouvellement des mots-clés entre la période récente et ancienne.\n"
        "Formule : 1 − (mots-clés communs / union des mots-clés)\n"
        "Interprétation :\n"
        "  ≈ 0.0 → vocabulaire stable\n"
        "  0.5   → moitié du vocabulaire a été renouvelé\n"
        "  ≈ 1.0 → vocabulaire entièrement différent\n"
        "Une dérive lexicale élevée sans dérive sémantique = changement de terminologie "
        "sans changement de sens.",
    ),
    "stability_score": (
        "Score de stabilité",
        "Inverse normalisé de la variance du volume mensuel. "
        "Mesure la régularité de la production d'articles.\n"
        "Interprétation :\n"
        "  ≈ 1.0 → volume très régulier mois après mois\n"
        "  ≈ 0.0 → volume très irrégulier (bursts et creux fréquents)\n"
        "Un topic avec Stabilité faible ET Burst actif est probablement driven by events "
        "(conférences, annonces majeures).",
    ),
    # Features enricher
    "mean_novelty": (
        "Nouveauté moyenne",
        "Score de nouveauté moyen des articles de ce topic (depuis l'enricher NLP). "
        "Mesure à quel point les articles sont sémantiquement éloignés des autres clusters.\n"
        "∈ [0, 1] : 1.0 = articles très originaux, non encore absorbés par les topics établis.",
    ),
    "mean_topic_prob": (
        "Prob. topic moyenne (BERTopic)",
        "Probabilité d'appartenance moyenne au topic BERTopic. "
        "Reflète la cohésion thématique interne du topic.\n"
        "∈ [0, 1] : proche de 1 → articles très homogènes autour du thème central.",
    ),
    "mean_centroid_distance": (
        "Distance centroïde moyenne",
        "Distance euclidienne moyenne de chaque article au centroïde de son topic "
        "(dans l'espace d'embedding).\n"
        "Faible → articles concentrés, topic précis · "
        "Élevée → topic diffus, articles thématiquement dispersés.",
    ),
    "pct_english": (
        "% articles en anglais",
        "Proportion d'articles rédigés en anglais dans ce topic.\n"
        "Très majoritairement en anglais dans arXiv. "
        "Un % inhabituellement bas peut signaler un topic émergent dans d'autres communautés.",
    ),
    "mean_flesch_grade": (
        "Lisibilité Flesch-Kincaid (niveau scolaire)",
        "Score Flesch-Kincaid moyen des abstracts, exprimé en niveau scolaire américain.\n"
        "Interprétation :\n"
        "  < 12 → texte accessible (lycée)\n"
        "  12–16 → niveau universitaire\n"
        "  > 16 → texte très technique (doctorat+)\n"
        "Un score élevé reflète des abstracts denses en jargon ou en formules.",
    ),
    "mean_latex_density": (
        "Densité LaTeX moyenne",
        "Proportion moyenne de tokens LaTeX (formules, symboles mathématiques) "
        "dans les abstracts de ce topic.\n"
        "Élevée → topic fortement mathématique ou théorique.",
    ),
    "pct_cross_category": (
        "% cross-catégorie",
        "Proportion d'articles appartenant simultanément à d'autres catégories arXiv.\n"
        "Élevée → topic interdisciplinaire, articles soumis dans plusieurs domaines.\n"
        "Peut indiquer un sujet à la frontière de plusieurs disciplines (ex : ML+Physics).",
    ),
}


# =========================================================================== #
# Point d'entrée public                                                        #
# =========================================================================== #

def render() -> None:
    """Point d'entrée de la Page 3 — appelé par app.py."""
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

    # Charger toutes les données du topic sélectionné en parallèle
    detail       = get_topic_detail(category_id, topic_id)
    monthly      = get_topic_monthly_stats(category_id, topic_id)
    full_history = get_trend_full_history(category_id, topic_id)

    if not detail:
        st.warning(
            "Impossible de charger les détails de ce topic. "
            "Les données de tendance sont peut-être incomplètes.",
            icon="⚠️",
        )
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


# =========================================================================== #
# Sélecteur topic                                                              #
# =========================================================================== #

def _render_topic_selector(
    topics: list[dict], category_id: str
) -> Optional[int]:
    """
    Sélecteur de topic pour la catégorie active.

    Réinitialise la sélection si la catégorie a changé depuis le dernier rendu.
    Retourne le topic_id sélectionné.
    """
    # Réinitialiser le topic sélectionné si la catégorie a changé
    if st.session_state.get("_topic_sel_category") != category_id:
        st.session_state["_topic_sel_category"] = category_id
        st.session_state.pop("selected_topic_id", None)

    topic_map = {t["topic_id"]: t for t in topics}
    options   = [t["topic_id"] for t in topics]

    current = st.session_state.get("selected_topic_id")
    if current not in options:
        st.session_state["selected_topic_id"] = options[0]

    def _fmt(tid: int) -> str:
        t = topic_map.get(tid, {})
        status = t.get("status") or ""
        icon   = _STATUS_ICONS.get(status, "○")
        label  = t.get("topic_label", f"Topic {tid}")
        score  = t.get("composite_score")
        score_str = f"  [{score:.3f}]" if score is not None else ""
        return f"{icon}  {label}{score_str}"

    return st.selectbox(
        "Topic",
        options=options,
        key="selected_topic_id",
        format_func=_fmt,
        label_visibility="collapsed",
    )


# =========================================================================== #
# Header                                                                       #
# =========================================================================== #

def _render_topic_header(detail: dict) -> None:
    """Affiche le nom du topic, son état et ses métriques clés."""
    status = detail.get("status") or "STABLE"
    color  = STATUS_COLORS.get(status, "#6B7280")
    label  = STATUS_LABELS.get(status, status)
    icon   = _STATUS_ICONS.get(status, "○")

    # Badge état inline avec le titre
    st.markdown(
        f"### {detail['topic_label']}\n"
        f"<span style='background:{color}22; color:{color}; border:1px solid {color}55; "
        f"border-radius:6px; padding:3px 10px; font-size:0.85em; font-weight:600;'>"
        f"{icon} {label}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Période : {detail.get('first_month', '—')} → {detail.get('last_month', '—')}  "
        f"·  {detail['n_months_active']} mois actif  "
        f"·  {detail['n_articles']:,} articles  ".replace(",", "\u202f")
        + f"·  Rapport du {detail.get('report_month', '—')}"
    )

    # KPIs principaux
    c1, c2, c3, c4 = st.columns(4)
    _metric_col(c1, "composite_score",    detail)
    _metric_col(c2, "trend_score",        detail)
    _metric_col(c3, "weak_signal_score",  detail)
    _metric_col(c4, "semantic_score",     detail)


def _metric_col(col, key: str, detail: dict) -> None:
    """Affiche une st.metric dans une colonne avec le label et le help issus de _METRIC_INFO."""
    name, help_text = _METRIC_INFO.get(key, (key, ""))
    val = detail.get(key)
    col.metric(
        label=name,
        value=f"{val:.4f}" if val is not None else "—",
        help=help_text,
    )


# =========================================================================== #
# Charts d'évolution                                                           #
# =========================================================================== #

def _render_evolution_charts(
    monthly: list[dict], full_history: list[dict]
) -> None:
    """Volume mensuel (bar) + scores composites (ligne) côte à côte."""
    st.subheader("Évolution temporelle")

    col_vol, col_score = st.columns(2)

    # ── Volume mensuel ────────────────────────────────────────────────────────
    with col_vol:
        st.caption("**Volume mensuel** (articles)")
        if not monthly:
            st.caption("Pas de données de volume.")
        else:
            months     = [d["month"] for d in monthly]
            n_articles = [d["n_articles"] for d in monthly]
            novelty    = [d["mean_novelty"] for d in monthly]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=months, y=n_articles,
                name="Articles",
                marker_color="#3B82F6",
                hovertemplate="<b>%{x}</b><br>%{y} articles<extra></extra>",
            ))
            # Superposer la nouveauté moyenne sur axe secondaire
            if any(v is not None for v in novelty):
                fig.add_trace(go.Scatter(
                    x=months, y=novelty,
                    name="Nouveauté moy.",
                    mode="lines+markers",
                    line=dict(color="#F59E0B", width=2),
                    yaxis="y2",
                    hovertemplate="<b>%{x}</b><br>Nouveauté : %{y:.3f}<extra></extra>",
                ))
                fig.update_layout(
                    yaxis2=dict(
                        overlaying="y", side="right",
                        showgrid=False, range=[0, 1],
                        tickfont=dict(color="#F59E0B"),
                    ),
                )
            fig.update_layout(**_chart_layout(height=260))
            st.plotly_chart(fig, use_container_width=True)

    # ── Scores dans le temps ──────────────────────────────────────────────────
    with col_score:
        st.caption("**Scores de tendance** dans le temps")
        if not full_history:
            st.caption("Pas d'historique de tendance.")
        else:
            months = [d["month"] for d in full_history]

            fig = go.Figure()
            traces = [
                ("composite_score",   "Score composite", "#3B82F6", 3),
                ("trend_score",       "Tendance",        "#00C49A", 2),
                ("weak_signal_score", "Signal faible",   "#8B5CF6", 2),
                ("semantic_score",    "Sémantique",      "#F59E0B", 2),
            ]
            for key, name, color, width in traces:
                vals = [d[key] for d in full_history]
                if any(v is not None for v in vals):
                    fig.add_trace(go.Scatter(
                        x=months, y=vals,
                        name=name,
                        mode="lines+markers",
                        line=dict(color=color, width=width),
                        hovertemplate=f"<b>%{{x}}</b><br>{name} : %{{y:.4f}}<extra></extra>",
                    ))
            fig.update_layout(
                **_chart_layout(height=260),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================================================================== #
# Grille des 13 métriques                                                      #
# =========================================================================== #

def _render_metrics_grid(detail: dict) -> None:
    """Affiche toutes les métriques du dernier rapport dans une grille 4 colonnes."""
    st.subheader("Métriques détaillées")
    st.caption(f"Rapport du mois : **{detail.get('report_month', '—')}**")

    # ── Métriques de tendance (exclut celles déjà dans le header) ─────────────
    trend_metrics = [
        "z_score_last",
        "ols_slope",
        "mann_kendall_tau",
        "mann_kendall_pvalue",
        "hurst_exponent",
        "cagr",
        "burst_state_last",
        "emergence_score_last",
        "novelty_weighted_emergence_last",
        "persistence_score",
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

    # ── Caractéristiques NLP des articles (features enricher) ─────────────────
    enricher_metrics = [
        ("mean_novelty",           "%.4f"),
        ("mean_topic_prob",        "%.4f"),
        ("mean_centroid_distance", "%.4f"),
        ("pct_english",            "%.1f %%"),
        ("mean_flesch_grade",      "%.2f"),
        ("mean_latex_density",     "%.4f"),
        ("pct_cross_category",     "%.1f %%"),
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
            display = (fmt % val) if val is not None else "—"
            col.metric(label=name, value=display, help=help_text)


# =========================================================================== #
# Chart de dérives                                                             #
# =========================================================================== #

def _render_drift_chart(full_history: list[dict]) -> None:
    """Line chart des dérives sémantique et lexicale dans le temps."""
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
    sem_drift  = [d["semantic_drift_last"] for d in full_history]
    kw_drift   = [d["keyword_drift_last"]  for d in full_history]
    stability  = [d["stability_score"]     for d in full_history]

    fig = go.Figure()

    if any(v is not None for v in sem_drift):
        fig.add_trace(go.Scatter(
            x=months, y=sem_drift,
            name="Dérive sémantique",
            mode="lines+markers",
            line=dict(color="#8B5CF6", width=2),
            hovertemplate="<b>%{x}</b><br>Dérive séma. : %{y:.4f}<extra></extra>",
        ))
    if any(v is not None for v in kw_drift):
        fig.add_trace(go.Scatter(
            x=months, y=kw_drift,
            name="Dérive lexicale",
            mode="lines+markers",
            line=dict(color="#EF4444", width=2),
            hovertemplate="<b>%{x}</b><br>Dérive lex. : %{y:.4f}<extra></extra>",
        ))
    if any(v is not None for v in stability):
        fig.add_trace(go.Scatter(
            x=months, y=stability,
            name="Stabilité",
            mode="lines",
            line=dict(color="#6B7280", width=1, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Stabilité : %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        **_chart_layout(height=260),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, font=dict(size=11),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "La **dérive sémantique** mesure l'évolution du sens (embeddings). "
        "La **dérive lexicale** mesure le renouvellement des mots-clés. "
        "Une valeur élevée indique un topic en transformation profonde."
    )


# =========================================================================== #
# Historique mensuel complet                                                   #
# =========================================================================== #

def _render_history_table(full_history: list[dict]) -> None:
    """Tableau de l'historique complet des scores par mois."""
    rows = []
    for d in full_history:
        status = d["status"] or "—"
        rows.append({
            "Mois":              d["month"],
            "État":              STATUS_LABELS.get(status, status),
            "Composite":         d["composite_score"],
            "Tendance":          d["trend_score"],
            "Signal faible":     d["weak_signal_score"],
            "Sémantique":        d["semantic_score"],
            "Z-score":           d["z_score_last"],
            "OLS slope":         d["ols_slope"],
            "CAGR":              d["cagr"],
            "Émergence":         d["emergence_score_last"],
            "Persistence":       d["persistence_score"],
            "Dérive séma.":      d["semantic_drift_last"],
            "Dérive lex.":       d["keyword_drift_last"],
            "Stabilité":         d["stability_score"],
        })

    df = pd.DataFrame(rows)
    float_cols = [c for c in df.columns if c not in ("Mois", "État")]
    col_config = {
        "Mois":  st.column_config.TextColumn("Mois", width="small"),
        "État":  st.column_config.TextColumn("État", width="medium"),
    }
    col_config.update({
        c: st.column_config.NumberColumn(c, format="%.4f", width="small")
        for c in float_cols
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        height=min(500, max(200, len(df) * 35 + 40)),
    )


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _chart_layout(height: int = 280) -> dict:
    """Paramètres Plotly communs pour tous les charts de la page."""
    return dict(
        margin=dict(l=0, r=0, t=10, b=0),
        height=height,
        xaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
