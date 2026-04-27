"""
Orchestre la Phase 1 : fit BERTopic sur le corpus complet d'une catégorie.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..storage.base_storage import BaseEnricherStorage, CleanPartition
from ..topic_modeling.topic_aligner import TopicAligner
from .nlp_pipeline import NlpPipeline


class TopicTrainer:
    """
    Entraîne BERTopic pour chaque catégorie sur le corpus complet de toutes les partitions CLEAN disponibles.

    Utilisé à la fois pour le fit initial (lancement du pipeline) et pour les refits périodiques (drift).
    """

    def __init__(
        self,
        pipeline: NlpPipeline,
        storage: BaseEnricherStorage,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._pipeline = pipeline
        self._storage = storage
        self._log = log or logging.getLogger(__name__)

    def fit_if_needed(self, partitions: list[CleanPartition]) -> None:
        """
        Entraîne BERTopic pour chaque catégorie qui en a besoin.
        """
        by_category: dict[str, list[CleanPartition]] = {}
        for p in partitions:
            by_category.setdefault(p.category, []).append(p)

        for category, cat_partitions in by_category.items():
            if not self._pipeline.topic_modeler.needs_fit(category):
                self._log.debug(
                    "Modèle BERTopic existant pour '%s' : fit ignoré.", category
                )
                continue
            self._fit_category(category, cat_partitions)

    def refit_window(
        self,
        category: str,
        all_partitions: list[CleanPartition],
    ) -> dict[int, int]:
        """
        Re-entraîne BERTopic sur la fenêtre glissante des N dernières partitions.

        Avant le fit, sauvegarde les anciens centroides. Après le fit, aligne
        les nouveaux topic_ids sur les anciens via TopicAligner (stabilité des IDs).
        """
        window = self._pipeline.topic_config.refit_window_months
        cat_parts = sorted(
            [p for p in all_partitions if p.category == category],
            key=lambda p: (p.year, p.month),
        )
        window_parts = cat_parts[-window:] if len(cat_parts) > window else cat_parts

        self._log.info(
            "Re-fit BERTopic (drift) : '%s' | fenêtre %d mois (%d disponibles).",
            category,
            window,
            len(window_parts),
        )

        # Charger les anciens centroides avant d'écraser le modèle
        old_centroids = self._pipeline.topic_modeler.get_centroids(category)

        self._fit_category(category, window_parts)

        # Aligner les nouveaux IDs sur les anciens (stabilité des series temporelles)
        new_centroids = self._pipeline.topic_modeler.get_centroids(category)
        if old_centroids is not None and new_centroids is not None:
            aligner = TopicAligner(similarity_threshold=0.75)
            mapping = aligner.align(old_centroids, new_centroids)
            if mapping:
                self._log.info(
                    "TopicAligner '%s' : %d topics alignés sur %d nouveaux.",
                    category,
                    len(mapping),
                    len(new_centroids),
                )
            return mapping

        return {}

    def _fit_category(
        self,
        category: str,
        partitions: list[CleanPartition],
    ) -> None:
        """
        Collecte le corpus complet d'une catégorie et entraîne BERTopic.
        """
        self._log.info(
            "Pré-fit BERTopic : category='%s' | %d partitions.",
            category,
            len(partitions),
        )

        all_texts: list[str] = []
        all_embeddings: list[np.ndarray] = []

        for partition in partitions:
            try:
                df = self._storage.read_clean(partition)
                if df.empty:
                    continue
                texts = self._pipeline.preprocessor.prepare_texts(df)
                embeddings = self._pipeline.embed(texts, partition)
                all_texts.extend(texts)
                all_embeddings.append(embeddings)
                self._log.debug(
                    "  %s/%d/%02d : %d articles collectés.",
                    partition.category,
                    partition.year,
                    partition.month,
                    len(texts),
                )
            except Exception as exc:
                self._log.warning(
                    "  Skip %s/%d/%02d : %s",
                    partition.category,
                    partition.year,
                    partition.month,
                    exc,
                )

        if not all_texts:
            self._log.warning(
                "  Aucun article collecté pour '%s' : fit ignoré.", category
            )
            return

        self._pipeline.topic_modeler.fit(
            all_texts,
            np.concatenate(all_embeddings, axis=0),
            category,
            n_partitions=len(partitions),
        )
