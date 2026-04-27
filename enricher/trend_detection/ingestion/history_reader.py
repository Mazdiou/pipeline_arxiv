"""
HistoryReader : lecture et validation de l'historique cumulatif (history.parquet).

Fait le lien entre TrendStorage (persistance) et TimeSeriesBuilder (construction
des séries temporelles). Valide le schéma, caste les types, et filtre par
plage de dates si nécessaire.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..storage.trend_storage import BaseTrendStorage

logger = logging.getLogger(__name__)

# Colonnes obligatoires pour construire les séries temporelles
_REQUIRED_COLUMNS = {"topic_id", "month", "n_articles"}

# Colonnes optionnelles avec leur type de cast et valeur par défaut
_OPTIONAL_COLUMNS: dict[str, tuple[str, object]] = {
    "pct_articles": ("float32", 0.0),
    "mean_novelty": ("float32", 0.0),
    "mean_centroid_distance": ("float32", 0.0),
    "mean_topic_prob": ("float32", 0.0),
    "mean_flesch_grade": ("float32", 0.0),
    "mean_latex_density": ("float32", 0.0),
    "pct_cross_category": ("float32", 0.0),
    "pct_english": ("float32", 100.0),
    "mean_lang_confidence": ("float32", 0.0),
    "dominant_lang": ("object", "en"),
    "top_keywords": ("object", None),
}


class HistoryReader:
    """
    Lit et valide l'historique cumulatif depuis le storage.

    Args:
        storage : Instance de BaseTrendStorage (local ou MinIO).
    """

    def __init__(self, storage: BaseTrendStorage) -> None:
        self._storage = storage

    def read(
        self,
        category: str,
        from_month: Optional[str] = None,
        to_month: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Lit l'historique d'une catégorie avec validation et cast des types.

        Args:
            category   : Catégorie arXiv (ex : "cs_AI").
            from_month : Mois de début "YYYY-MM" (inclus). None = depuis le début.
            to_month   : Mois de fin "YYYY-MM" (inclus). None = jusqu'au dernier.

        Returns:
            DataFrame validé et typé. Vide si aucun historique disponible.
        """
        df = self._storage.read_history(category)

        if df.empty:
            logger.debug("Aucun historique disponible pour '%s'.", category)
            return df

        # Vérification des colonnes obligatoires
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.error(
                "Historique '%s' invalide : colonnes manquantes %s.", category, missing
            )
            return pd.DataFrame()

        # Cast des colonnes obligatoires
        df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").astype("Int32")
        df["n_articles"] = (
            pd.to_numeric(df["n_articles"], errors="coerce").fillna(0).astype("int32")
        )
        df["month"] = df["month"].astype(str)

        # Cast et remplissage des colonnes optionnelles
        for col, (dtype, default) in _OPTIONAL_COLUMNS.items():
            if col not in df.columns:
                df[col] = default
            elif dtype == "float32":
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(default)
                    .astype("float32")
                )

        # Filtre par plage de dates
        if from_month:
            df = df[df["month"] >= from_month]
        if to_month:
            df = df[df["month"] <= to_month]

        df = df.sort_values(["topic_id", "month"], ignore_index=True)

        logger.debug(
            "Historique '%s' lu : %d entrées, %d topics, %d mois.",
            category,
            len(df),
            df["topic_id"].nunique(),
            df["month"].nunique(),
        )
        return df
