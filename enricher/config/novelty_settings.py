"""
Configuration du module de calcul de scores de nouveauté/similarité intra-partition.

intra_similarity = moyenne cosinus des K voisins les plus similaires.
novelty_score    = 1 - intra_similarity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoveltyConfig:
    """
    Paramètres du module ArticleNoveltyScorer.
    """

    knn_k: int = 5
    """
    Nombre de plus proches voisins pour le calcul de similarité intra-partition.
    """

    enabled: bool = True
    """
    False : scores mis à 0.0 sans calcul.
    """

    min_articles: int = 12
    """
    Nombre minimum d'articles dans le lot pour calculer les scores KNN.
    En dessous : intra_similarity=0.0, novelty_score=1.0, centroid_distance=0.0.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.knn_k < 1:
            raise ValueError(f"'knn_k' doit être >= 1 (reçu : {self.knn_k})")
        if self.min_articles < 1:
            raise ValueError(
                f"'min_articles' doit être >= 1 (reçu : {self.min_articles})"
            )
