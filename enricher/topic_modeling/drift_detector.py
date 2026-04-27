"""
DriftDetector : detecte si le modele BERTopic courant a derive.
"""

from __future__ import annotations

from ..config.topic_settings import TopicConfig


class DriftDetector:
    """
    Detecte si le modele BERTopic doit etre re-entraine.
    """

    def __init__(self, config: TopicConfig) -> None:
        self._threshold = config.refit_outlier_threshold

    def should_refit(self, outlier_rate: float) -> bool:
        """
        Retourne True si le taux d'outliers depasse le seuil.
        """
        return outlier_rate > self._threshold

    @property
    def threshold(self) -> float:
        return self._threshold
