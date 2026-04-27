"""
TopicResult : encapsule les résultats du topic modeling pour un batch d'articles.
- topic_ids    : topic_id par article (-1 pour les outliers).
- topic_labels : label textuel par article (ex: 'neural_network_attention').
- topic_probs  : probabilité d'appartenance [0.0, 1.0] par article.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TopicResult:
    """
    Résultat du topic modeling pour un batch d'articles.
    """

    topic_ids: list[int]
    topic_labels: list[str]
    topic_probs: list[float]

    def __len__(self) -> int:
        return len(self.topic_ids)

    def to_dataframe(self, index: Optional[pd.Index] = None) -> pd.DataFrame:
        """
        Convertit le résultat en DataFrame avec les colonnes GOLD attendues.
        """
        df = pd.DataFrame(
            {
                "topic_id": pd.array(self.topic_ids, dtype="int32"),
                "topic_label": self.topic_labels,
                "topic_prob": pd.array(self.topic_probs, dtype="float32"),
                "is_topic_outlier": [tid == -1 for tid in self.topic_ids],
            }
        )
        if index is not None:
            df.index = index
        return df
