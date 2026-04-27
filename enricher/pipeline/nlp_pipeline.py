"""
Pipeline NLP spécialisé pour transformer une seule partition 
de CLEAN en GOLD via des étapes séquentielles.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config.settings import EnricherConfig
from ..embeddings.embedder import ArticleEmbedder
from ..keyword_extraction.keyword_extractor import ArticleKeywordExtractor
from ..lang_detection.lang_detector import LangDetector
from ..ner.ner_extractor import ArticleNerExtractor
from ..novelty.novelty_scorer import ArticleNoveltyScorer
from ..preprocessing.text_preprocessor import NlpTextPreprocessor
from ..quality.quality_analyzer import ArticleQualityAnalyzer
from ..storage.base_storage import CleanPartition
from ..topic_modeling.topic_modeler import ArticleTopicModeler
from .embedding_cache import EmbeddingCache


class NlpPipeline:
    """
    Pipeline NLP pour transformer un seul DataFrame CLEAN en GOLD.
    """

    def __init__(
        self,
        config: EnricherConfig,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._cfg = config
        self._log = log or logging.getLogger(__name__)

        self._embedder = ArticleEmbedder(config.embeddings)
        self._preprocessor = NlpTextPreprocessor(config.preprocessing)
        self._lang_detector = LangDetector(config.lang)
        self._topic_modeler = ArticleTopicModeler(config.topic)
        self._keyword_extractor = ArticleKeywordExtractor(config.keyword)
        self._ner_extractor = ArticleNerExtractor(config.ner)
        self._novelty_scorer = ArticleNoveltyScorer(config.novelty)
        self._quality_analyzer = ArticleQualityAnalyzer(config.quality)
        self._cache = EmbeddingCache(
            cache_dir=config._embeddings_cache_dir,
            model_name=config.embeddings.model_name,
        )
        self._model_injected = False

    # Injection du modèle de SentenceTransformer dans les composants qui en ont besoin
    def _ensure_model_injected(self) -> None:
        """
        Injecte le SentenceTransformer dans TopicModeler et KeywordExtractor.
        """
        if self._model_injected:
            return
        model = self._embedder.sentence_model
        self._topic_modeler._sentence_model = model
        self._keyword_extractor._sentence_model = model
        self._model_injected = True

    # Proprietes exposees pour TopicTrainer
    @property
    def topic_modeler(self) -> ArticleTopicModeler:
        return self._topic_modeler

    @property
    def preprocessor(self) -> NlpTextPreprocessor:
        return self._preprocessor

    @property
    def topic_config(self):
        """Expose la TopicConfig pour TopicTrainer (évite l'accès à _cfg)."""
        return self._cfg.topic

    def embed(
        self,
        texts: list[str],
        partition: Optional[CleanPartition] = None,
    ) -> np.ndarray:
        """Calcule ou recupere depuis le cache disque les embeddings d'un batch."""
        self._ensure_model_injected()
        return self._cache.get_or_compute(texts, partition, self._embedder)

    # Pipeline principal (etapes 1 a 9)
    def run(
        self,
        df: pd.DataFrame,
        category: str,
        partition: Optional[CleanPartition] = None,
    ) -> pd.DataFrame:
        """
        Applique les 9 etapes NLP sur un DataFrame CLEAN.
        """
        self._ensure_model_injected()

        idx = df.index
        self._log.info("Pipeline NLP : %d articles | category='%s'", len(df), category)

        # 1. Pretraitement
        nlp_texts = self._preprocessor.prepare_texts(df)
        lang_texts = self._preprocessor.prepare_lang_texts(df)
        latex_densities = self._preprocessor.compute_latex_density(df)

        # 2. Embeddings
        embeddings = self._cache.get_or_compute(nlp_texts, partition, self._embedder)

        # 3. Langue
        lang_result = self._lang_detector.detect(lang_texts)

        # 4. Topic modeling
        topic_result = self._topic_modeler.assign_topics(
            nlp_texts, embeddings, category
        )

        # 5. Mots-cles
        keyword_result = self._keyword_extractor.extract(
            nlp_texts, embeddings=embeddings
        )

        # 6. NER
        raw_texts = (
            df["title"].fillna("").astype(str)
            + " "
            + df["abstract"].fillna("").astype(str)
        ).tolist()
        ner_result = self._ner_extractor.extract(raw_texts)

        # 7. Nouveaute
        novelty_result = self._novelty_scorer.score(embeddings)

        # 8. Qualite
        quality_result = self._quality_analyzer.analyze(df, latex_densities)

        # 9. Assemblage GOLD
        df_gold = df.copy()
        for result in (
            lang_result,
            topic_result,
            keyword_result,
            ner_result,
            novelty_result,
            quality_result,
        ):
            for col, series in result.to_dataframe(index=idx).items():
                df_gold[col] = series

        if self._cfg.embeddings.store_in_gold:
            df_gold["embedding"] = [e.tolist() for e in embeddings]

        self._log_gold_stats(df_gold)

        return df_gold

    # Helper prive
    def _log_gold_stats(self, df_gold: pd.DataFrame) -> None:
        """Log les statistiques cles du DataFrame GOLD assemble."""
        n = len(df_gold)
        if "is_topic_outlier" in df_gold.columns:
            n_out = int(df_gold["is_topic_outlier"].sum())
            self._log.info(
                "  Outliers BERTopic : %d/%d (%.1f%%)",
                n_out,
                n,
                100.0 * n_out / n if n > 0 else 0.0,
            )
        if "lang" in df_gold.columns:
            n_non_en = int((df_gold["lang"] != "en").sum())
            if n_non_en > 0:
                self._log.info(
                    "  Articles non-anglais : %d/%d (%.1f%%)",
                    n_non_en,
                    n,
                    100.0 * n_non_en / n if n > 0 else 0.0,
                )
