from .time_series_builder import TimeSeriesBuilder, TopicTimeSeries
from .trend_metrics import (
    compute_cagr,
    compute_hurst_exponent,
    compute_mann_kendall,
    compute_ols_slope,
    compute_z_score_series,
)
from .weak_signal_metrics import (
    compute_burst_score,
    compute_emergence_score,
    compute_novelty_weighted_emergence,
    compute_persistence_score,
)
from .semantic_drift import (
    compute_keyword_drift,
    compute_semantic_drift,
    compute_stability_score,
)

__all__ = [
    # Séries temporelles
    "TimeSeriesBuilder",
    "TopicTimeSeries",
    # Métriques de tendance
    "compute_z_score_series",
    "compute_ols_slope",
    "compute_mann_kendall",
    "compute_hurst_exponent",
    "compute_cagr",
    # Métriques de signal faible
    "compute_emergence_score",
    "compute_novelty_weighted_emergence",
    "compute_persistence_score",
    "compute_burst_score",
    # Métriques sémantiques
    "compute_semantic_drift",
    "compute_keyword_drift",
    "compute_stability_score",
]
