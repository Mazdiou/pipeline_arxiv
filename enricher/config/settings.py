"""
Configuration principale du module enricher.
EnricherConfig compose toutes les sous-configurations par composant NLP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .embeddings_settings import EmbeddingsConfig
from .keyword_settings import KeywordConfig
from .lang_settings import LangConfig
from .ner_settings import NerConfig
from .novelty_settings import NoveltyConfig
from .preprocessing_settings import PreprocessingConfig
from .quality_settings import QualityConfig
from .topic_settings import TopicConfig
from .trend_settings import TrendConfig


@dataclass
class EnricherConfig:
    """
    Configuration complète du module ArxivEnricher.
    """

    # Sous-configs par composant NLP
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    lang: LangConfig = field(default_factory=LangConfig)
    topic: TopicConfig = field(default_factory=TopicConfig)
    keyword: KeywordConfig = field(default_factory=KeywordConfig)
    ner: NerConfig = field(default_factory=NerConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)

    # Chemins de stockage
    clean_path: str = "data/CLEAN"
    """Chemin racine de la zone CLEAN (source)."""

    gold_path: str = "data/GOLD"
    """Chemin racine de la zone GOLD (destination)."""

    trends_path: str = "data/TRENDS"
    """Chemin racine de la zone TRENDS (statistiques par topic)."""

    models_path: str = "models"
    """
    Dossier racine pour les modèles BERTopic et les caches d'embeddings.
    """

    # Filtres de traitement
    categories: Optional[List[str]] = None
    """
    Liste de catégories à traiter. None : toutes les catégories disponibles dans CLEAN.
    Exemple : ['cs.AI', 'stat.ML']
    """

    years: Optional[List[int]] = None
    """Années à traiter. None : toutes les années disponibles."""

    months: Optional[List[int]] = None
    """Mois à traiter (1-12). None : tous les mois disponibles."""

    # Options de run
    skip_existing: bool = True
    """
    True : ignorer les partitions GOLD déjà existantes (mode incrémental).
    False : réenrichir toutes les partitions (mode full-refresh).
    """

    storage_backend: str = field(
        default_factory=lambda: os.getenv("STORAGE_BACKEND", "local")
    )
    """Backend de stockage : 'local' | 'minio'. Lu depuis STORAGE_BACKEND dans .env."""

    def __post_init__(self) -> None:
        """
        Synchronise models_path avec les chemins internes des sous-configs
        et valide les paramètres de niveau pipeline.
        """
        if self.storage_backend not in ("local", "minio"):
            raise ValueError(
                f"'storage_backend' invalide : '{self.storage_backend}'. "
                "Valeurs acceptées : 'local', 'minio'."
            )

        # Dérive un nom de dossier safe depuis le modèle d'embeddings
        _safe_model = (
            self.embeddings.model_name.replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )

        self.topic.models_dir = f"{self.models_path}/topics/{_safe_model}"
        self.embeddings.cache_dir = f"{self.models_path}/sentence_transformers"
        self._embeddings_cache_dir = f"{self.models_path}/embeddings_cache"
        # Synchronise TrendConfig avec les chemins et le modèle configurés

        self.trend.embeddings_cache_path = self._embeddings_cache_dir
        self.trend.embeddings_model_name = self.embeddings.model_name
        self.trend.gold_data_path = self.gold_path
