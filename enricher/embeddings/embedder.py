"""
Génération d'embeddings sémantiques pour les articles arXiv.

Les embeddings sont calculés UNE SEULE FOIS par partition.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config.embeddings_settings import EmbeddingsConfig

logger = logging.getLogger(__name__)


class ArticleEmbedder:
    """
    Génère des embeddings sémantiques denses pour les textes d'articles.
    """

    def __init__(self, config: Optional[EmbeddingsConfig] = None) -> None:
        self._cfg = config or EmbeddingsConfig()
        self._model = None
        self._embedding_dim: Optional[int] = None

    # Propriétés
    @property
    def sentence_model(self):
        """
        Modèle SentenceTransformer chargé, prêt à l'emploi.
        """
        self._ensure_model()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """
        Dimension des vecteurs produits par ce modèle.
        """
        if self._embedding_dim is None:
            self._ensure_model()
        return self._embedding_dim

    # API publique
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Génère les embeddings pour une liste de textes préprocessés.
        """
        self._ensure_model()

        if not texts:
            assert self._embedding_dim is not None
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        n_empty = sum(1 for t in texts if not t.strip())
        if n_empty:
            logger.warning(
                "%d texte(s) vide(s) dans le batch — "
                "les embeddings correspondants auront un faible signal.",
                n_empty,
            )

        logger.debug(
            "Embedding : %d textes | modèle=%s | batch=%d | device=%s",
            len(texts),
            self._cfg.model_name,
            self._cfg.batch_size,
            self._cfg.resolved_device(),
        )

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=self._cfg.normalize_embeddings,
            show_progress_bar=self._cfg.show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embeddings_to_series(
        self,
        embeddings: np.ndarray,
        index: Optional[pd.Index] = None,
    ) -> pd.Series:
        """
        Convertit la matrice d'embeddings en pd.Series pour le GOLD Parquet.
        """
        rows = [row.tolist() for row in embeddings]
        return pd.Series(rows, index=index, name="embedding", dtype=object)

    # Méthodes internes
    def _ensure_model(self) -> None:
        """
        Charge le modèle SentenceTransformer si ce n'est pas encore fait.
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Le module 'sentence-transformers' est requis pour ArticleEmbedder.\n"
                "Installation : pip install sentence-transformers"
            ) from exc

        device = self._cfg.resolved_device()
        logger.info(
            "Chargement SentenceTransformer : modèle='%s' | device='%s'",
            self._cfg.model_name,
            device,
        )

        self._model = SentenceTransformer(
            self._cfg.model_name,
            device=device,
            cache_folder=self._cfg.cache_dir,
        )

        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Modèle prêt — dimension=%d | device='%s'",
            self._embedding_dim,
            device,
        )
