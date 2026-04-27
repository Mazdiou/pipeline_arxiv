"""
SignalScorer : calcul des scores de tendance, signal faible et transformation sémantique.

Ce module transforme un ensemble de métriques hétérogènes (statistiques, temporelles,
sémantiques) en scores normalisés et interprétables.

Pipeline global :
    1. Normalisation des métriques (sigmoïde ou mise à l'échelle)
    2. Agrégation par dimension (trend / weak signal / semantic)
    3. Combinaison pondérée en score composite

Important :
    - Les scores sont conçus pour être majoritairement dans [0, 1]
    - Certaines métriques peuvent légèrement dépasser 1 si non bornées (ex: burst)
    - Le score composite est renormalisé pour rester stable
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Data containers
@dataclass
class TopicScores:
    """
    Contient les scores calculés pour un topic.

    Tous les scores sont des agrégations normalisées et comparables entre topics.
    """

    trend_score: float = 0.0
    """Force de la tendance volumétrique (croissance + régularité + dynamique)."""

    weak_signal_score: float = 0.0
    """Intensité d'un signal émergent ou en phase de détection précoce."""

    semantic_score: float = 0.0
    """Niveau de transformation sémantique (évolution du contenu / vocabulaire)."""

    composite_score: float = 0.0
    """Score final pondéré combinant les 3 dimensions."""


@dataclass(frozen=True)
class ScorerWeights:
    """
    Pondérations des dimensions dans le score composite.

    Les poids n'ont pas besoin de sommer à 1 :
    une normalisation est appliquée automatiquement.
    """

    alpha: float = 0.4  # trend
    beta: float = 0.4  # weak signal
    gamma: float = 0.2  # semantic


# Core scorer
class SignalScorer:
    """
    Classe principale de scoring.

    Transforme des métriques brutes en scores interprétables.
    Chaque dimension est calculée indépendamment puis combinée.
    """

    def __init__(self, weights: ScorerWeights = ScorerWeights()) -> None:
        self._w = weights

    # Public API
    def score(
        self,
        z_score_last: float,
        ols_slope: float,
        mann_kendall_tau: float,
        hurst_exponent: float,
        cagr: float,
        burst_state_last: int,
        burst_max_states: int,
        emergence_score_last: float,
        novelty_weighted_emergence_last: float,
        persistence_score: float,
        semantic_drift_last: float,
        keyword_drift_last: float,
        stability_score: float,
    ) -> TopicScores:
        """
        Calcule les scores dimensionnels et le score composite.

        Toutes les entrées sont supposées être calculées en amont.
        Cette fonction ne fait que normaliser + agréger.

        Returns:
            TopicScores : scores arrondis pour lisibilité (pas pour calcul interne).
        """

        trend_score = self._compute_trend_score(
            z_score_last, ols_slope, mann_kendall_tau, hurst_exponent, cagr
        )

        weak_signal_score = self._compute_weak_signal_score(
            burst_state_last,
            burst_max_states,
            emergence_score_last,
            novelty_weighted_emergence_last,
            persistence_score,
        )

        semantic_score = self._compute_semantic_score(
            semantic_drift_last, keyword_drift_last, stability_score
        )

        composite_score = self._compute_composite(
            trend_score, weak_signal_score, semantic_score
        )

        return TopicScores(
            trend_score=round(trend_score, 4),
            weak_signal_score=round(weak_signal_score, 4),
            semantic_score=round(semantic_score, 4),
            composite_score=round(composite_score, 4),
        )

    # Dimension computations
    def _compute_trend_score(
        self,
        z_score: float,
        ols_slope: float,
        mk_tau: float,
        hurst: float,
        cagr: float,
    ) -> float:
        """
        Score de tendance volumétrique.

        Combine :
            - Intensité (z-score)
            - Vitesse (slope)
            - Direction (Mann-Kendall)
            - Structure temporelle (Hurst)
            - Croissance globale (CAGR)

        Chaque composant est ramené dans une échelle comparable.
        """

        z_norm = _sigmoid(_safe(z_score), scale=1.5)
        slope_norm = _sigmoid(_safe(ols_slope), scale=0.5)
        tau_norm = (_safe(mk_tau) + 1.0) / 2.0
        hurst_norm = _clamp(_safe(hurst), 0.0, 1.0)
        cagr_norm = _sigmoid(_safe(cagr), scale=2.0)

        return (z_norm + slope_norm + tau_norm + hurst_norm + cagr_norm) / 5.0

    def _compute_weak_signal_score(
        self,
        burst_state: int,
        burst_max_states: int,
        emergence: float,
        nwe: float,
        persistence: float,
    ) -> float:
        """
        Score de signal faible.

        Combine :
            - Burst (activité soudaine)
            - Emergence (croissance récente)
            - NWE (émergence pondérée par nouveauté)
            - Persistence (continuité du signal)

        Note :
            burst peut dépasser 1 si mal calibré en amont : clamp léger.
        """

        burst_norm = burst_state / max(burst_max_states, 1)
        burst_norm = _clamp(burst_norm, 0.0, 1.5)

        emergence_norm = _sigmoid(_safe(emergence), scale=3.0)
        nwe_norm = _sigmoid(_safe(nwe), scale=3.0)
        persistence_norm = _clamp(_safe(persistence), 0.0, 1.0)

        return (burst_norm + emergence_norm + nwe_norm + persistence_norm) / 4.0

    def _compute_semantic_score(
        self,
        drift: float,
        kw_drift: float,
        stability: float,
    ) -> float:
        """
        Score de transformation sémantique.

        Interprétation :
            - drift élevé : changement global du contenu
            - keyword drift : évolution du vocabulaire
            - stability inversée : instabilité = changement

        Attention :
            Ce score mesure le CHANGEMENT, pas la qualité.
        """

        instability = 1.0 - _clamp(_safe(stability), 0.0, 1.0)

        drift_norm = _clamp(_safe(drift), 0.0, 1.0)
        kw_drift_norm = _clamp(_safe(kw_drift), 0.0, 1.0)

        return (drift_norm + kw_drift_norm + instability) / 3.0

    def _compute_composite(
        self,
        trend: float,
        weak_signal: float,
        semantic: float,
    ) -> float:
        """
        Combine les dimensions avec pondération.

        Garantit un score stable même si les poids changent.
        """

        w = self._w
        total_weight = w.alpha + w.beta + w.gamma

        if total_weight < 1e-10:
            return 0.0

        composite = w.alpha * trend + w.beta * weak_signal + w.gamma * semantic

        return composite / total_weight


# Helpers
def _sigmoid(x: float, scale: float = 1.0) -> float:
    """
    Fonction sigmoïde stable numériquement.

    Utilisée pour transformer des valeurs non bornées en score ∈ (0, 1).
    """
    try:
        return 1.0 / (1.0 + math.exp(-scale * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _safe(x: float, default: float = 0.0) -> float:
    """
    Nettoie une valeur numérique.

    Évite les NaN / inf qui cassent les scores.
    """
    if x is None or math.isnan(x) or math.isinf(x):
        return default
    return x


def _clamp(x: float, min_val: float, max_val: float) -> float:
    """
    Contraint une valeur dans un intervalle donné.
    """
    return max(min(x, max_val), min_val)
