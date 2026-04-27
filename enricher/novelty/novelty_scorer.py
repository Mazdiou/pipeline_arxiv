"""
Calcul des scores de nouveauté et de similarité intra-partition.

intra_similarity  : similarité cosinus moyenne avec les K plus proches voisins.
novelty_score     : 1 - intra_similarity.
centroid_distance : distance cosinus au centroïde de la partition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config.novelty_settings import NoveltyConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoveltyResult:
    """
    Résultat du calcul de nouveauté pour une partition d'articles.
    """

    intra_similarity: list
    novelty_score: list
    centroid_distance: list

    def __len__(self) -> int:
        return len(self.novelty_score)

    def to_dataframe(self, index: Optional[Any] = None) -> pd.DataFrame:
        """
        Convertit le résultat en DataFrame avec 3 colonnes float32.
        """
        return pd.DataFrame(
            {
                "intra_similarity": pd.array(self.intra_similarity, dtype="float32"),
                "novelty_score": pd.array(self.novelty_score, dtype="float32"),
                "centroid_distance": pd.array(self.centroid_distance, dtype="float32"),
            },
            index=index,
        )


class ArticleNoveltyScorer:
    """
    Calcule les scores de nouveauté et similarité intra-partition pour chaque article.
    """

    def __init__(self, config: Optional[NoveltyConfig] = None) -> None:
        """
        Args:
            config: Configuration du scorer.
        """
        self._cfg = config or NoveltyConfig()

    def score(self, embeddings: np.ndarray) -> NoveltyResult:
        """
        Calcule les 3 scores de nouveauté pour une partition d'articles.
        """
        n = len(embeddings)

        if not self._cfg.enabled:
            logger.debug("NoveltyScorer désactivé.")
            return NoveltyResult(
                intra_similarity=[0.0] * n,
                novelty_score=[0.0] * n,
                centroid_distance=[0.0] * n,
            )

        if n < self._cfg.min_articles:
            logger.warning(
                "Corpus trop petit pour novelty (%d < %d) : novelty_score=1.0.",
                n,
                self._cfg.min_articles,
            )
            return NoveltyResult(
                intra_similarity=[0.0] * n,
                novelty_score=[1.0] * n,
                centroid_distance=[0.0] * n,
            )

        sim_matrix = _cosine_similarity_matrix(embeddings)
        intra_sim = _knn_similarity(sim_matrix, k=self._cfg.knn_k)
        novelty = (1.0 - intra_sim).clip(0.0, 1.0).astype(np.float32)
        centroid_dist = _centroid_distance(embeddings)

        logger.debug(
            "NoveltyScorer : n=%d | novelty_mean=%.3f | centroid_mean=%.3f",
            n,
            float(novelty.mean()),
            float(centroid_dist.mean()),
        )

        return NoveltyResult(
            intra_similarity=intra_sim.tolist(),
            novelty_score=novelty.tolist(),
            centroid_distance=centroid_dist.tolist(),
        )


# Fonctions utilitaires privées
def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de similarité cosinus (n, n).
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(embeddings)
    except ImportError:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.where(norms > 0, norms, 1.0)
        return np.dot(normed, normed.T)


def _knn_similarity(sim_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Calcule la similarité moyenne aux K plus proches voisins pour chaque article.
    """
    n = sim_matrix.shape[0]
    k_eff = min(k, n - 1)  # on ne peut pas dépasser n-1 voisins

    if k_eff <= 0:
        return np.zeros(n, dtype=np.float32)

    sim_no_self = sim_matrix.copy()
    np.fill_diagonal(sim_no_self, -1.0)  # exclure la self-similarity

    # Top-K voisins (K plus grandes similarités après exclusion de la diagonale)
    top_k_sims = np.sort(sim_no_self, axis=1)[:, -k_eff:]
    return top_k_sims.mean(axis=1).clip(0.0, 1.0).astype(np.float32)


def _centroid_distance(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcule la distance cosinus de chaque article au centroïde de la partition.

    centroid_distance = 1 - cosine_similarity(article, centroid_normalisé)
    """
    centroid = embeddings.mean(axis=0, keepdims=True)  # (1, dim)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    dot_products = (embeddings * centroid).sum(axis=1)  # (n,)
    return (1.0 - dot_products).clip(0.0, 1.0).astype(np.float32)
