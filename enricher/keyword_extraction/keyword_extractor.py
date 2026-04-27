"""
Extraction de mots-clés scientifiques avec KeyBERT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config.keyword_settings import KeywordConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeywordResult:
    """
    Résultat de l'extraction de mots-clés pour une liste d'articles.
    """

    keywords: list
    scores: list

    def __len__(self) -> int:
        return len(self.keywords)

    def to_dataframe(self, index: Optional[Any] = None) -> pd.DataFrame:
        """
        Convertit le résultat en DataFrame.
        """
        pairs = [list(zip(kws, scs)) for kws, scs in zip(self.keywords, self.scores)]
        return pd.DataFrame(
            {"keywords": pd.array(pairs, dtype=object)},
            index=index,
        )


class ArticleKeywordExtractor:
    """
    Extrait des mots-clés pertinents et diversifiés par article avec KeyBERT.
    """

    def __init__(
        self,
        config: Optional[KeywordConfig] = None,
        sentence_model: Any = None,
    ) -> None:
        """
        Args:
            config         : Configuration KeyBERT.
            sentence_model : Modèle SentenceTransformer partagé depuis ArticleEmbedder.
        """
        self._cfg = config or KeywordConfig()
        self._sentence_model: Any = sentence_model
        self._kw_model: Any = None

    # API publique
    def extract(
        self,
        texts: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> KeywordResult:
        """
        Extrait les mots-clés pour une liste de textes preprocessés.

        Returns:
            KeywordResult(keywords, scores) : list de mots-clés et scores correspondants par article.
        """
        self._ensure_model()

        if not texts:
            return KeywordResult(keywords=[], scores=[])

        raw = self._extract_batch(texts, embeddings)
        keywords = [[kw for kw, _ in kws] for kws in raw]
        scores = [[float(s) for _, s in kws] for kws in raw]
        return KeywordResult(keywords=keywords, scores=scores)

    # Méthodes internes
    def _ensure_model(self) -> None:
        """Charge KeyBERT en lazy avec le modèle SentenceTransformer partagé."""
        if self._kw_model is not None:
            return
        try:
            from keybert import KeyBERT
        except ImportError as exc:
            raise ImportError(
                "keybert est requis pour l'extraction de mots-clés. "
                "Installation : pip install keybert"
            ) from exc

        if self._sentence_model is not None:
            self._kw_model = KeyBERT(model=self._sentence_model)
            logger.debug(
                "KeyBERT initialisé avec le modèle SentenceTransformer partagé."
            )
        else:
            self._kw_model = KeyBERT()
            logger.debug(
                "KeyBERT initialisé avec son propre modèle (double chargement)."
            )

    def _extract_batch(
        self,
        texts: list[str],
        embeddings: Optional[np.ndarray],
    ) -> list[list[tuple]]:
        """
        Extrait les mots-clés en batch. Fallback article-par-article si erreur.

        Returns:
            list[list[tuple[str, float]]] : mots-clés et scores par article.
        """

        kwargs = self._build_kwargs(embeddings)
        try:
            results = self._kw_model.extract_keywords(texts, **kwargs)

            # KeyBERT retourne list[list[tuple]] pour plusieurs textes
            if isinstance(results, list) and results and isinstance(results[0], list):
                return results

            # Pour un seul texte, KeyBERT peut retourner list[tuple] directement
            return [results]
        except Exception as exc:
            logger.warning(
                "Extraction batch échouée (%s) : fallback article-par-article.", exc
            )
            return self._extract_fallback(texts, kwargs)

    def _extract_fallback(
        self,
        texts: list[str],
        kwargs: dict,
    ) -> list[list[tuple]]:
        """Extraction article-par-article en cas d'échec du batch."""

        results: list[list[tuple]] = []
        for i, text in enumerate(texts):
            single_kwargs = {k: v for k, v in kwargs.items() if k != "doc_embeddings"}
            try:
                kws = self._kw_model.extract_keywords([text], **single_kwargs)
                results.append(kws[0] if kws else [])
            except Exception as exc:
                logger.debug("Extraction échouée pour article %d : %s", i, exc)
                results.append([])
        return results

    def _build_kwargs(self, embeddings: Optional[np.ndarray]) -> dict:
        """Construit les paramètres d'appel à KeyBERT.extract_keywords()."""
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            ngram_range=self._cfg.keyphrase_ngram_range,
            stop_words=self._cfg.stop_words,
            min_df=self._cfg.min_df,
        )
        kwargs: dict = {
            "top_n": self._cfg.top_n,
            "vectorizer": vectorizer,
        }
        if self._cfg.use_mmr:
            kwargs["use_mmr"] = True
            kwargs["diversity"] = self._cfg.diversity
        if embeddings is not None:
            kwargs["doc_embeddings"] = embeddings
        return kwargs
