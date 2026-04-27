"""
EmbeddingAggregator — calcule les centroids sémantiques par (topic_id, mois).

Principe :
    Les partitions GOLD et le cache d'embeddings sont ALIGNÉS par partition :
    la ligne i du GOLD correspond à la ligne i du fichier .npy du même mois.

    Pour chaque mois :
        1. Lire GOLD/{category}/{year}/{month}.parquet : topic_id par article
        2. Lire embeddings_cache/{model}/{category}/{year}_{month}.npy : vecteurs (n, 768)
        3. Grouper les vecteurs par topic_id : calculer la moyenne (centroïde)

    Résultat : dict[topic_id] : dict[mois] : centroïde (np.ndarray 768-dim)
    Utilisé par semantic_drift.compute_semantic_drift() pour le module DST.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _make_minio_gold_reader(category: str) -> Callable[[str, str], pd.DataFrame]:
    """
    Retourne un callable qui lit une partition GOLD depuis MinIO.

    Le callable prend (year: str, month_num: str) et retourne un DataFrame
    avec au moins la colonne topic_id, ou un DataFrame vide si absent.
    """
    from minio_config import MINIO_CONFIG
    from minio import Minio

    client = Minio(
        MINIO_CONFIG.endpoint,
        access_key=MINIO_CONFIG.access_key,
        secret_key=MINIO_CONFIG.secret_key,
        secure=MINIO_CONFIG.secure,
    )
    bucket = MINIO_CONFIG.gold_bucket

    def _read(year: str, month_num: str) -> pd.DataFrame:
        key = f"{category}/{year}/{month_num}/articles.parquet"
        response = None
        try:
            response = client.get_object(bucket, key)
            data = response.read()
            return pd.read_parquet(io.BytesIO(data), columns=["topic_id"])
        except Exception:
            logger.debug("GOLD absent dans MinIO : %s/%s", bucket, key)
            return pd.DataFrame()
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    return _read


class EmbeddingAggregator:
    """
    Calcule les centroids sémantiques par (topic_id, mois) pour le DST.

    Args:
        gold_path             : Chemin racine des partitions GOLD locales (ex : "data/GOLD").
                                Ignoré si storage_backend="minio".
        embeddings_cache_path : Chemin racine du cache embeddings (toujours local).
        model_name            : Nom du modèle (sous-dossier dans le cache).
        storage_backend       : "local" ou "minio". Défaut : variable STORAGE_BACKEND.
    """

    def __init__(
        self,
        gold_path: str,
        embeddings_cache_path: str,
        model_name: str = "allenai-specter",
        storage_backend: Optional[str] = None,
    ) -> None:
        self._gold = Path(gold_path)
        self._cache = Path(embeddings_cache_path)
        # Normalisation du nom de modèle pour le chemin de fichier
        self._model = model_name.replace("/", "_").replace("-", "_").lower()
        self._storage_backend = storage_backend or os.getenv("STORAGE_BACKEND", "local")
        # Le reader MinIO est initialisé à la demande (lazy, par catégorie)
        self._minio_readers: dict[str, Callable[[str, str], pd.DataFrame]] = {}

    def compute_centroids(
        self,
        category: str,
        months: list[str],
    ) -> dict[int, dict[str, np.ndarray]]:
        """
        Calcule les centroids par topic et par mois pour une catégorie.

        Args:
            category : Catégorie arXiv (ex : "cs_AI").
            months   : Liste de mois "YYYY-MM" à traiter.

        Returns:
            dict[topic_id] : dict["YYYY-MM"] : centroïde (np.ndarray 768-dim).
            Un topic absent d'un mois n'a pas d'entrée pour ce mois.
        """
        # Résultat structuré : topic_id : {mois : centroïde}
        centroids: dict[int, dict[str, np.ndarray]] = {}

        for month in months:
            month_centroids = self._compute_month_centroids(category, month)
            for topic_id, centroid in month_centroids.items():
                if topic_id not in centroids:
                    centroids[topic_id] = {}
                centroids[topic_id][month] = centroid

        total = sum(len(v) for v in centroids.values())
        logger.debug(
            "EmbeddingAggregator : %d topics, %d centroids calculés pour '%s'.",
            len(centroids),
            total,
            category,
        )
        return centroids

    def _compute_month_centroids(
        self, category: str, month: str
    ) -> dict[int, np.ndarray]:
        """
        Calcule les centroids pour tous les topics d'un mois donné.

        Returns:
            dict[topic_id] : centroïde. Vide si les données sont absentes.
        """
        year, month_num = month.split("-")

        cache_path = self._cache / self._model / category / f"{year}_{month_num}.npy"
        if not cache_path.exists():
            logger.debug("Cache embeddings absent : %s", cache_path)
            return {}

        try:
            if self._storage_backend == "minio":
                if category not in self._minio_readers:
                    self._minio_readers[category] = _make_minio_gold_reader(category)
                gold_df = self._minio_readers[category](year, month_num)
                if gold_df.empty:
                    return {}
            else:
                gold_path = (
                    self._gold / category / year / month_num / "articles.parquet"
                )
                if not gold_path.exists():
                    logger.debug("GOLD absent : %s", gold_path)
                    return {}
                gold_df = pd.read_parquet(gold_path, columns=["topic_id"])

            embeddings = np.load(cache_path)
        except Exception as exc:
            logger.warning("Erreur lecture %s/%s : %s", category, month, exc)
            return {}

        # Vérification de l'alignement GOLD ↔ embeddings
        if len(gold_df) != len(embeddings):
            logger.warning(
                "Désalignement GOLD/embeddings pour %s/%s : %d vs %d lignes.",
                category,
                month,
                len(gold_df),
                len(embeddings),
            )
            return {}

        result: dict[int, np.ndarray] = {}

        for topic_id, group in gold_df.groupby("topic_id"):
            tid = int(topic_id)
            if tid == -1:
                continue  # Ignorer les outliers BERTopic

            # Les indices du groupe correspondent aux lignes du cache
            topic_embeddings = embeddings[group.index.to_numpy()]
            # Centroïde = moyenne des vecteurs : représentation sémantique du topic ce mois
            centroid = topic_embeddings.mean(axis=0).astype(np.float32)
            result[tid] = centroid

        return result
