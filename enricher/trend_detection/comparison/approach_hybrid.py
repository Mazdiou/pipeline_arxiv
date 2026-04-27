"""
Module de l'approche Hybride (fusion statistique + sémantique).

Calcule les 13 métriques (6 de tendance, 4 de signal faible, 3 sémantiques)
et produit un score composite pondéré + classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..classification.topic_classifier import (
    ClassifierThresholds,
    TopicClassifier,
    TopicStatus,
)
from ..classification.signal_scorer import ScorerWeights, SignalScorer, TopicScores
from ..core.semantic_drift import (
    compute_keyword_drift,
    compute_semantic_drift,
    compute_stability_score,
)
from ..core.time_series_builder import TopicTimeSeries
from ..core.trend_metrics import (
    compute_cagr,
    compute_hurst_exponent,
    compute_mann_kendall,
    compute_ols_slope,
    compute_z_score_series,
)
from ..core.weak_signal_metrics import (
    compute_burst_score,
    compute_emergence_score,
    compute_novelty_weighted_emergence,
    compute_persistence_score,
)


@dataclass
class HybridResult:
    """Résultat complet d'analyse d'un topic."""

    topic_id: int

    # Trend metrics
    z_score_last: float
    ols_slope: float
    mann_kendall_tau: float
    mann_kendall_pvalue: float
    hurst_exponent: float
    cagr: float

    # Weak signals
    burst_state_last: int
    emergence_score_last: float
    novelty_weighted_emergence_last: float
    persistence_score: float

    # Semantic signals
    semantic_drift_last: float
    keyword_drift_last: float
    stability_score: float

    # Outputs
    status: TopicStatus
    scores: TopicScores

    # Time series (dashboard)
    months: list[str]
    n_articles_series: list[int]
    z_score_series: list[float]
    burst_series: list[int]
    semantic_drift_series: list[float]


class ApproachHybrid:
    """
    Args:
        z_score_window    : Fenêtre glissante pour le z-score (mois).
        hurst_min_len     : Longueur minimale de la série pour Hurst.
        burst_max_states  : États max pour Kleinberg.
        weights           : Pondérations α, β, γ du score composite.
        thresholds        : Seuils de classification (ClassifierThresholds).
    """

    def __init__(
        self,
        z_score_window: int = 4,
        hurst_min_len: int = 8,
        burst_max_states: int = 2,
        weights: ScorerWeights = ScorerWeights(),
        thresholds: ClassifierThresholds = ClassifierThresholds(),
    ) -> None:
        self._z_window = z_score_window
        self._hurst_min = hurst_min_len
        self._burst_states = burst_max_states
        self._scorer = SignalScorer(weights)
        self._classifier = TopicClassifier(thresholds)

    def analyze(
        self,
        ts: TopicTimeSeries,
        centroids_by_month: dict[str, np.ndarray] | None = None,
    ) -> HybridResult:
        """
        Analyse complète d'une série temporelle (toutes métriques).
        """
        # Série complète (incluant mois à 0) pour les métriques de tendance et signal faible.
        vol = ts.volume_series

        # Série filtrée (mois actifs uniquement) pour les métriques sensibles aux zéros.
        vol_active = [v for v in vol if v > 0]

        # Métriques de tendance
        z_series = compute_z_score_series(vol, self._z_window)
        ols_slope = compute_ols_slope(vol)
        mk_tau, mk_pvalue = compute_mann_kendall(vol_active)
        hurst = compute_hurst_exponent(vol_active, self._hurst_min)
        cagr = compute_cagr(vol)

        # Métriques de signal faible
        # Emergence et NWE : série COMPLÈTE (vol, avec zéros).
        # Burst : série COMPLÈTE (vol) car on veut détecter les pics même après des mois à 0.
        # Persistence : série ACTIVE (vol_active) car on veut mesurer la continuité des mois actifs.
        burst_series = compute_burst_score(vol, self._burst_states)
        emergence_series = compute_emergence_score(vol)
        nwe_series = compute_novelty_weighted_emergence(vol, ts.mean_novelty)
        persistence = compute_persistence_score(vol_active)

        # Métriques sémantiques
        if centroids_by_month and len(centroids_by_month) >= 2:
            drift_by_month = compute_semantic_drift(centroids_by_month)
            drift_series = [drift_by_month.get(m, 0.0) for m in ts.months]
        else:
            drift_series = [0.0] * len(ts.months)

        keywords_by_month = {
            m: kws for m, kws in zip(ts.months, ts.top_keywords_by_month) if kws
        }
        if len(keywords_by_month) >= 2:
            kw_drift_by_month = compute_keyword_drift(keywords_by_month)
            kw_drift_last = (
                kw_drift_by_month.get(ts.months[-1], 0.0) if ts.months else 0.0
            )
        else:
            kw_drift_last = 0.0

        stability = compute_stability_score([d for d in drift_series if d > 0])

        # Valeurs du dernier mois
        z_last = z_series[-1] if z_series else 0.0
        burst_last = burst_series[-1] if burst_series else 0
        emergence_last = emergence_series[-1] if emergence_series else 0.0
        nwe_last = nwe_series[-1] if nwe_series else 0.0
        drift_last = drift_series[-1] if drift_series else 0.0
        n_last = ts.n_articles[-1] if ts.n_articles else 0
        pct_last = ts.pct_articles[-1] if ts.pct_articles else 100.0

        # Score composite
        scores = self._scorer.score(
            z_score_last=z_last,
            ols_slope=ols_slope,
            mann_kendall_tau=mk_tau,
            hurst_exponent=hurst,
            cagr=cagr,
            burst_state_last=burst_last,
            burst_max_states=self._burst_states,
            emergence_score_last=emergence_last,
            novelty_weighted_emergence_last=nwe_last,
            persistence_score=persistence,
            semantic_drift_last=drift_last,
            keyword_drift_last=kw_drift_last,
            stability_score=stability,
        )

        # Classification
        status = self._classifier.classify(
            n_months_active=ts.n_months_active,
            n_articles_last=n_last,
            mann_kendall_tau=mk_tau,
            mann_kendall_pvalue=mk_pvalue,
            hurst_exponent=hurst,
            emergence_score_last=emergence_last,
            semantic_drift_last=drift_last,
            pct_articles_last=pct_last,
        )

        return HybridResult(
            topic_id=ts.topic_id,
            z_score_last=z_last,
            ols_slope=ols_slope,
            mann_kendall_tau=mk_tau,
            mann_kendall_pvalue=mk_pvalue,
            hurst_exponent=hurst,
            cagr=cagr,
            burst_state_last=burst_last,
            emergence_score_last=emergence_last,
            novelty_weighted_emergence_last=nwe_last,
            persistence_score=persistence,
            semantic_drift_last=drift_last,
            keyword_drift_last=kw_drift_last,
            stability_score=stability,
            status=status,
            scores=scores,
            months=ts.months,
            n_articles_series=ts.n_articles,
            z_score_series=z_series,
            burst_series=burst_series,
            semantic_drift_series=drift_series,
        )
