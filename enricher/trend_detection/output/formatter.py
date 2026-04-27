"""
TrendFormatter : sérialise les résultats de l'approche Hybride pour le dashboard.

Produit deux formats :
    - JSON     : pour l'API / dashboard Streamlit (dict sérialisable)
    - Parquet  : pour stockage et chargement rapide (pandas DataFrame)

Structure de sortie par catégorie et par mois de run :
    CategoryTrendReport
    └── topics: list[TopicTrendRecord]
            ├── identité         : topic_id, topic_label
            ├── statut           : status (6 états)
            ├── scores           : trend_score, weak_signal_score, semantic_score, composite_score
            ├── métriques clés   : z_score, mann_kendall_tau, hurst, cagr, emergence, nwe, persistence
            ├── sémantique       : semantic_drift, keyword_drift, stability_score
            ├── features enricher: mean_novelty, mean_topic_prob, mean_centroid_distance,
            │                      pct_cross_category, mean_flesch_grade, mean_latex_density, pct_english
            └── séries           : months, n_articles_series, z_score_series,
                                   burst_series, semantic_drift_series
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..classification.topic_classifier import TopicStatus
from ..comparison.approach_hybrid import HybridResult
from ..core.time_series_builder import TopicTimeSeries


@dataclass
class TopicTrendRecord:
    """Enregistrement complet des tendances pour un topic."""

    # Identité
    topic_id: int
    topic_label: str

    # Statut et scores
    status: str
    trend_score: float
    weak_signal_score: float
    semantic_score: float
    composite_score: float

    # Métriques statistiques de tendance
    z_score_last: float
    ols_slope: float
    mann_kendall_tau: float
    mann_kendall_pvalue: float
    hurst_exponent: float
    cagr: float

    # Métriques de signal faible
    burst_state_last: int
    emergence_score_last: float
    novelty_weighted_emergence_last: float
    persistence_score: float

    # Métriques sémantiques (DST)
    semantic_drift_last: float
    keyword_drift_last: float
    stability_score: float

    # Features héritées de l'enricher (depuis history.parquet)
    mean_novelty: float
    mean_topic_prob: float
    mean_centroid_distance: float
    pct_cross_category: float
    mean_flesch_grade: float
    mean_latex_density: float
    pct_english: float

    # Séries temporelles pour visualisation
    months: list[str] = field(default_factory=list)
    n_articles_series: list[int] = field(default_factory=list)
    z_score_series: list[float] = field(default_factory=list)
    burst_series: list[int] = field(default_factory=list)
    semantic_drift_series: list[float] = field(default_factory=list)


@dataclass
class CategoryTrendReport:
    """Rapport de tendances pour une catégorie arXiv sur un mois donné."""

    category: str
    """Catégorie arXiv (ex : 'cs_AI')."""

    period: str
    """Mois de run au format 'YYYY-MM'."""

    n_topics_total: int
    """Nombre total de topics analysés."""

    n_trending: int
    """Nombre de topics en état TRENDING."""

    n_emerging: int
    """Nombre de topics en état EMERGING (signaux faibles)."""

    n_transforming: int
    """Nombre de topics en état TRANSFORMING."""

    n_declining: int
    """Nombre de topics en état DECLINING."""

    topics: list[TopicTrendRecord] = field(default_factory=list)
    """Liste des enregistrements par topic, triés par composite_score décroissant."""


class TrendFormatter:
    """
    Sérialise les résultats HybridResult en formats exploitables par le dashboard.

    Usage :
        formatter = TrendFormatter()
        report = formatter.build_report(category, period, results, time_series_map)
        formatter.to_json(report, "output/trends_cs_AI_2025-03.json")
        formatter.to_parquet(report, "output/trends_cs_AI_2025-03.parquet")
    """

    def build_report(
        self,
        category: str,
        period: str,
        hybrid_results: dict[int, HybridResult],
        time_series_map: dict[int, TopicTimeSeries],
        topic_labels: dict[int, str] | None = None,
    ) -> CategoryTrendReport:
        """
        Construit le rapport de tendances pour une catégorie.

        Args:
            category        : Catégorie arXiv (ex : "cs_AI").
            period          : Mois de run "YYYY-MM".
            hybrid_results  : dict[topic_id] : HybridResult (depuis ApproachHybrid).
            time_series_map : dict[topic_id] : TopicTimeSeries (pour les features enricher).
            topic_labels    : dict[topic_id] : label (optionnel, complète les labels manquants).

        Returns:
            CategoryTrendReport prêt à être sérialisé.
        """
        records: list[TopicTrendRecord] = []

        for topic_id, result in hybrid_results.items():
            ts = time_series_map.get(topic_id)
            label = (topic_labels or {}).get(topic_id, f"topic_{topic_id}")

            # Features héritées de l'enricher : dernière valeur disponible dans la série
            mean_novelty = ts.mean_novelty[-1] if ts and ts.mean_novelty else 0.0
            mean_topic_prob = (
                ts.mean_topic_prob[-1] if ts and ts.mean_topic_prob else 0.0
            )
            mean_centroid_dist = (
                ts.mean_centroid_distance[-1]
                if ts and ts.mean_centroid_distance
                else 0.0
            )
            pct_cross = (
                ts.pct_cross_category[-1] if ts and ts.pct_cross_category else 0.0
            )
            flesch = ts.mean_flesch_grade[-1] if ts and ts.mean_flesch_grade else 0.0
            latex = ts.mean_latex_density[-1] if ts and ts.mean_latex_density else 0.0
            pct_en = ts.pct_english[-1] if ts and ts.pct_english else 100.0

            records.append(
                TopicTrendRecord(
                    topic_id=topic_id,
                    topic_label=label,
                    status=result.status.value,
                    trend_score=result.scores.trend_score,
                    weak_signal_score=result.scores.weak_signal_score,
                    semantic_score=result.scores.semantic_score,
                    composite_score=result.scores.composite_score,
                    z_score_last=result.z_score_last,
                    ols_slope=result.ols_slope,
                    mann_kendall_tau=result.mann_kendall_tau,
                    mann_kendall_pvalue=result.mann_kendall_pvalue,
                    hurst_exponent=result.hurst_exponent,
                    cagr=result.cagr,
                    burst_state_last=result.burst_state_last,
                    emergence_score_last=result.emergence_score_last,
                    novelty_weighted_emergence_last=result.novelty_weighted_emergence_last,
                    persistence_score=result.persistence_score,
                    semantic_drift_last=result.semantic_drift_last,
                    keyword_drift_last=result.keyword_drift_last,
                    stability_score=result.stability_score,
                    mean_novelty=round(float(mean_novelty), 4),
                    mean_topic_prob=round(float(mean_topic_prob), 4),
                    mean_centroid_distance=round(float(mean_centroid_dist), 4),
                    pct_cross_category=round(float(pct_cross), 4),
                    mean_flesch_grade=round(float(flesch), 4),
                    mean_latex_density=round(float(latex), 4),
                    pct_english=round(float(pct_en), 4),
                    months=result.months,
                    n_articles_series=result.n_articles_series,
                    z_score_series=result.z_score_series,
                    burst_series=result.burst_series,
                    semantic_drift_series=result.semantic_drift_series,
                )
            )

        # Trier par composite_score décroissant
        records.sort(key=lambda r: r.composite_score, reverse=True)

        statuses = [r.status for r in records]
        return CategoryTrendReport(
            category=category,
            period=period,
            n_topics_total=len(records),
            n_trending=statuses.count(TopicStatus.TRENDING.value),
            n_emerging=statuses.count(TopicStatus.EMERGING.value),
            n_transforming=statuses.count(TopicStatus.TRANSFORMING.value),
            n_declining=statuses.count(TopicStatus.DECLINING.value),
            topics=records,
        )

    def to_json(self, report: CategoryTrendReport, path: str) -> None:
        """
        Sérialise le rapport en JSON.

        Args:
            report : CategoryTrendReport à sérialiser.
            path   : Chemin de sortie (ex : "output/trends_cs_AI_2025-03.json").
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = _report_to_dict(report)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

    def to_dataframe(self, report: CategoryTrendReport) -> pd.DataFrame:
        """Convertit un CategoryTrendReport en DataFrame (une ligne par topic, séries exclues)."""
        if not report.topics:
            return pd.DataFrame()

        rows = []
        for r in report.topics:
            rows.append(
                {
                    "category": report.category,
                    "period": report.period,
                    "topic_id": r.topic_id,
                    "topic_label": r.topic_label,
                    "status": r.status,
                    "composite_score": r.composite_score,
                    "trend_score": r.trend_score,
                    "weak_signal_score": r.weak_signal_score,
                    "semantic_score": r.semantic_score,
                    "z_score_last": r.z_score_last,
                    "ols_slope": r.ols_slope,
                    "mann_kendall_tau": r.mann_kendall_tau,
                    "mann_kendall_pvalue": r.mann_kendall_pvalue,
                    "hurst_exponent": r.hurst_exponent,
                    "cagr": r.cagr,
                    "burst_state_last": r.burst_state_last,
                    "emergence_score_last": r.emergence_score_last,
                    "novelty_weighted_emergence_last": r.novelty_weighted_emergence_last,
                    "persistence_score": r.persistence_score,
                    "semantic_drift_last": r.semantic_drift_last,
                    "keyword_drift_last": r.keyword_drift_last,
                    "stability_score": r.stability_score,
                    "mean_novelty": r.mean_novelty,
                    "mean_topic_prob": r.mean_topic_prob,
                    "mean_centroid_distance": r.mean_centroid_distance,
                    "pct_cross_category": r.pct_cross_category,
                    "mean_flesch_grade": r.mean_flesch_grade,
                    "mean_latex_density": r.mean_latex_density,
                    "pct_english": r.pct_english,
                }
            )
        return pd.DataFrame(rows)

    def to_parquet(self, report: CategoryTrendReport, path: str) -> None:
        """
        Sérialise le rapport en Parquet local (une ligne par topic, séries exclues).

        Args:
            report : CategoryTrendReport à sérialiser.
            path   : Chemin de sortie local.
        """
        df = self.to_dataframe(report)
        if df.empty:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")


# Helpers
def _report_to_dict(report: CategoryTrendReport) -> dict[str, Any]:
    """Convertit CategoryTrendReport en dict JSON-sérialisable."""
    return asdict(report)


class _NumpyEncoder(json.JSONEncoder):
    """
    Encodeur JSON qui convertit les types NumPy en types Python natifs.

    Nécessaire car les séries temporelles (burst_series, n_articles_series)
    peuvent contenir des numpy.int64 / numpy.float32 issus des calculs vectorisés.
    """

    def default(self, obj: Any) -> Any:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
