"""
TrendConfig configuration complète du module trend_detection.
"""

from __future__ import annotations
from .embeddings_settings import EmbeddingsConfig

from dataclasses import dataclass


@dataclass
class TrendConfig:
    """Configuration du module trend_detection."""

    enabled: bool = True
    """False : snapshot non calculé (désactive tout le module)."""

    # Snapshot
    top_keywords_per_topic: int = 15
    """
    Nombre de mots-clés dominants à conserver par topic dans le snapshot.
    """

    min_articles_per_topic: int = 5
    """Seuil minimum d'articles pour qu'un topic soit inclus dans le snapshot."""

    # Métriques de tendance
    min_months_for_trend: int = 6
    """
    Nombre minimum de mois d'historique pour calculer les métriques de tendance.
    """

    z_score_window: int = 4
    """
    Taille de la fenêtre glissante (en mois) pour le z-score.
    """

    hurst_min_length: int = 12
    """
    Longueur minimale de la série pour calculer l'exposant de Hurst (méthode R/S).
    """

    # Signaux faibles
    burst_max_states: int = 2
    """
    Nombre d'états de burst pour l'algorithme de Kleinberg.
    2 = binaire (burst / non-burst).
    """

    # Dérive sémantique (DST)
    semantic_drift_enabled: bool = True
    """
    Active le calcul de la dérive sémantique (Dynamic Semantic Trajectory).
    Désactiver si les embeddings ne sont pas disponibles.
    """

    embeddings_cache_path: str = "models/embeddings_cache"
    """Chemin vers le dossier contenant les embeddings en cache (.npy par partition)."""

    embeddings_model_name: str = EmbeddingsConfig.model_name
    """Nom du modèle d'embeddings (sous-dossier dans embeddings_cache_path)."""

    gold_data_path: str = "data/GOLD"
    """Chemin vers les partitions GOLD (pour lier topic_id et embeddings)."""

    def __post_init__(self) -> None:
        if self.top_keywords_per_topic < 1:
            raise ValueError("top_keywords_per_topic doit être >= 1.")
        if self.min_articles_per_topic < 1:
            raise ValueError("min_articles_per_topic doit être >= 1.")
        if self.min_months_for_trend < 2:
            raise ValueError("min_months_for_trend doit être >= 2.")
        if self.z_score_window < 2:
            raise ValueError("z_score_window doit être >= 2.")
        if self.hurst_min_length < 4:
            raise ValueError("hurst_min_length doit être >= 4.")
        if self.burst_max_states < 1:
            raise ValueError("burst_max_states doit être >= 1.")
