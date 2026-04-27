"""
TopicModeler : gère la création et l'application de modèles de topic modeling (BERTopic) pour les articles.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Optional

import numpy as np

from ..config.topic_settings import TopicConfig
from .bertopic_factory import create_bertopic_model
from .model_store import TopicModelStore
from .topic_labels import extract_labels
from .outlier_reducer import reduce_outliers
from .topic_probabilities import all_outliers, build_topic_probs
from .topic_result import TopicResult

logger = logging.getLogger(__name__)


class ArticleTopicModeler:

    def __init__(self, config: TopicConfig, sentence_model: Any = None):
        self._cfg = config
        self._sentence_model = sentence_model
        self._store = TopicModelStore(config.models_dir)

        self._models: dict[str, Any] = {}

    def needs_fit(self, category: str) -> bool:
        if not self._cfg.enabled:
            return False
        return self._cfg.force_refit or not self._store.model_exists(category)

    def fit(self, texts, embeddings, category, n_partitions=0):
        if not self._cfg.enabled:
            return

        n = len(texts)

        if n < self._cfg.min_articles:
            logger.warning("Corpus trop petit pour fit")
            return

        model = create_bertopic_model(
            self._effective_config(n),
            self._sentence_model,
        )

        topics_raw, _ = model.fit_transform(texts, embeddings=embeddings)

        topics_raw = [int(t) for t in topics_raw]

        topics_final, _ = reduce_outliers(
            model,
            topics_raw,
            embeddings,
            threshold=self._cfg.outlier_threshold,
        )

        if topics_final != topics_raw:
            model.update_topics(texts, topics=topics_final)

        valid = [t for t in topics_final if t != -1]
        max_topic_id = max(valid) if valid else -1
        n_topics = len(set(valid))

        self._store.save_model(model, category)

        # Sauvegarder les centroides pour TopicAligner (re-fit futur)
        centroids = getattr(model, "topic_embeddings_", None)
        if centroids is not None:
            self._store.save_centroids(category, np.array(centroids))

        self._store.update_category_state(
            category,
            max_topic_id=max_topic_id,
            n_partitions=n_partitions,
        )

        logger.info(
            "Fit terminé pour '%s' : %d topics, max_topic_id=%d, outliers=%d/%d.",
            category,
            n_topics,
            max_topic_id,
            topics_final.count(-1),
            n,
        )

        self._models[category] = model

    def assign_topics(self, texts, embeddings, category) -> TopicResult:
        n = len(texts)

        if not self._cfg.enabled:
            return all_outliers(n)

        model = self._get_model(category)

        if model is None:
            return all_outliers(n)

        topics_raw, probs_raw = model.transform(texts, embeddings=embeddings)

        topics_raw = [int(t) for t in topics_raw]

        topics_final, outlier_sims = reduce_outliers(
            model,
            topics_raw,
            embeddings,
            threshold=self._cfg.outlier_threshold,
        )

        outlier_rate = topics_final.count(-1) / n if n > 0 else 0.0
        self._store.record_outlier_rate(category, outlier_rate)

        return TopicResult(
            topic_ids=topics_final,
            topic_labels=extract_labels(model, topics_final, texts),
            topic_probs=build_topic_probs(
                probs_raw, topics_raw, topics_final, outlier_sims
            ),
        )

    def get_centroids(self, category: str):
        """
        Retourne les centroides du modele courant pour la categorie, ou None.

        Utilise model.topic_embeddings_ si le modele est en memoire,
        sinon charge depuis le disque via model_store.
        """
        model = self._models.get(category)
        if model is not None:
            centroids = getattr(model, "topic_embeddings_", None)
            if centroids is not None:
                return np.array(centroids)
        return self._store.load_centroids(category)

    def _get_model(self, category) -> Optional[Any]:
        if category in self._models:
            return self._models[category]

        if not self._store.model_exists(category):
            return None

        model = self._store.load_model(
            category,
            sentence_model=self._sentence_model,
        )

        self._models[category] = model

        return model

    def _effective_config(self, n: int) -> TopicConfig:
        min_cluster = self._cfg.hdbscan_min_cluster_size
        n_neighbors = self._cfg.umap_n_neighbors

        if self._cfg.scale_cluster_size:
            min_cluster = max(min_cluster, min(50, n // self._cfg.cluster_size_divisor))

        if self._cfg.scale_n_neighbors:
            n_neighbors = max(n_neighbors, min(50, n // self._cfg.n_neighbors_divisor))

        min_samples = min(self._cfg.hdbscan_min_samples, min_cluster)

        if (
            min_cluster == self._cfg.hdbscan_min_cluster_size
            and n_neighbors == self._cfg.umap_n_neighbors
        ):
            return self._cfg

        return replace(
            self._cfg,
            hdbscan_min_cluster_size=min_cluster,
            hdbscan_min_samples=min_samples,
            umap_n_neighbors=n_neighbors,
        )
