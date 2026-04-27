"""
SnapshotResult : statistiques aggregees par topic pour une partition GOLD.

14 champs (un element par topic) :
  Identité        : topic_id, topic_label
  Volume          : n_articles, pct_articles
  Nouveauté       : mean_novelty
  Mots-clés       : top_keywords
  Cohérence topic : mean_centroid_distance, mean_topic_prob
  Lisibilité      : mean_flesch_grade
  Langue          : dominant_lang, pct_english, mean_lang_confidence
  Interdiscipl.   : pct_cross_category
  LaTeX           : mean_latex_density
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Colonnes incluses dans l'historique cumulatif (history.parquet)
_HISTORY_COLUMNS = [
    "topic_id",
    "month",
    "n_articles",
    "pct_articles",
    "mean_novelty",
    "top_keywords",
    "mean_centroid_distance",
    "mean_flesch_grade",
    "dominant_lang",
    "pct_english",
    "mean_topic_prob",
    "pct_cross_category",
    "mean_latex_density",
    "mean_lang_confidence",
]


@dataclass(frozen=True)
class SnapshotResult:
    """Statistiques aggregees par topic pour une partition GOLD."""

    # Identité
    topic_id: list[int] = field(default_factory=list)
    topic_label: list[str] = field(default_factory=list)

    # Volume
    n_articles: list[int] = field(default_factory=list)
    pct_articles: list[float] = field(default_factory=list)

    # Nouveauté
    mean_novelty: list[float] = field(default_factory=list)

    # Mots-clés
    top_keywords: list[list[str]] = field(default_factory=list)

    # Cohérence du topic
    mean_centroid_distance: list[float] = field(default_factory=list)
    mean_topic_prob: list[float] = field(default_factory=list)

    # Lisibilité
    mean_flesch_grade: list[float] = field(default_factory=list)

    # Langue
    dominant_lang: list[str] = field(default_factory=list)
    pct_english: list[float] = field(default_factory=list)
    mean_lang_confidence: list[float] = field(default_factory=list)

    # Interdisciplinarité
    pct_cross_category: list[float] = field(default_factory=list)

    # LaTeX
    mean_latex_density: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.topic_id)

    def to_dataframe(self, index: Optional[pd.Index] = None) -> pd.DataFrame:
        """Convertit en DataFrame. Vide si len == 0."""
        if len(self) == 0:
            return pd.DataFrame(columns=list(self.__dataclass_fields__.keys()))
        df = pd.DataFrame({k: v for k, v in self.__dict__.items()})
        if index is not None:
            df.index = index
        return df

    def to_history_dataframe(self, month: str) -> pd.DataFrame:
        """
        Convertit en DataFrame pour stockage dans TrendStorage (history.parquet).

        Inclut la colonne 'month' et toutes les métriques agrégées.
        Les snapshots antérieurs lus depuis le disque peuvent manquer
        les nouvelles colonnes — TimeSeriesBuilder utilise .get() avec
        des valeurs neutres pour la rétro-compatibilité.

        Args:
            month : Clé temporelle au format 'YYYY-MM'.

        Returns:
            DataFrame avec les colonnes définies dans _HISTORY_COLUMNS.
            Vide si len == 0.
        """
        if len(self) == 0:
            return pd.DataFrame(columns=_HISTORY_COLUMNS)

        return pd.DataFrame(
            {
                "topic_id": self.topic_id,
                "month": month,
                "n_articles": self.n_articles,
                "pct_articles": self.pct_articles,
                "mean_novelty": self.mean_novelty,
                "top_keywords": self.top_keywords,
                "mean_centroid_distance": self.mean_centroid_distance,
                "mean_flesch_grade": self.mean_flesch_grade,
                "dominant_lang": self.dominant_lang,
                "pct_english": self.pct_english,
                "mean_topic_prob": self.mean_topic_prob,
                "pct_cross_category": self.pct_cross_category,
                "mean_latex_density": self.mean_latex_density,
                "mean_lang_confidence": self.mean_lang_confidence,
            }
        )
