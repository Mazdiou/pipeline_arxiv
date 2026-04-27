"""
Cache disque des embeddings par partition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..embeddings.embedder import ArticleEmbedder
from ..storage.base_storage import CleanPartition

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache disque d'embeddings float32 par partition (fichiers .npy)."""

    def __init__(self, cache_dir: str, model_name: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._safe_model = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )

    def get_or_compute(
        self,
        texts: list[str],
        partition: Optional[CleanPartition],
        embedder: ArticleEmbedder,
    ) -> np.ndarray:
        """
        Retourne les embeddings float32 depuis le cache ou les calcule.

        Validation : si le cache contient un nombre de lignes différent de
        len(texts), il est invalidé et les embeddings sont recalculés.
        """
        if partition is None:
            return embedder.embed(texts).astype(np.float32)

        n_expected = len(texts)
        path = self._cache_path(partition)

        if path.exists():
            try:
                cached = np.load(str(path))
                n_cached, dim_cached = cached.shape[0], (
                    cached.shape[1] if cached.ndim > 1 else -1
                )
                dim_expected = embedder.embedding_dim
                if n_cached == n_expected and dim_cached == dim_expected:
                    return cached.astype(np.float32)
                if n_cached != n_expected:
                    logger.warning(
                        "Cache invalidé pour %s : %d lignes en cache vs %d attendues → recalcul.",
                        partition,
                        n_cached,
                        n_expected,
                    )
                else:
                    logger.warning(
                        "Cache invalidé pour %s : dim=%d en cache vs dim=%d attendu "
                        "(changement de modèle ?) → recalcul.",
                        partition,
                        dim_cached,
                        dim_expected,
                    )
            except Exception as exc:
                logger.warning("Cache read échoué (%s) → recalcul.", exc)

        embeddings = embedder.embed(texts).astype(np.float32)
        self._save(path, embeddings)
        return embeddings

    def _save(self, path: Path, embeddings: np.ndarray) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(path), embeddings)
        except Exception as exc:
            logger.warning("Cache save échoué : %s.", exc)

    def _cache_path(self, partition: CleanPartition) -> Path:
        safe_cat = partition.category.replace(".", "_").replace("/", "_")
        return (
            self._cache_dir
            / self._safe_model
            / safe_cat
            / f"{partition.year}_{partition.month:02d}.npy"
        )
