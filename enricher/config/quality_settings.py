"""
Configuration du module d'analyse de qualité NLP des articles arXiv.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityConfig:
    """
    Paramètres du module ArticleQualityAnalyzer.
    """

    short_abstract_threshold: int = 50
    """
    Nombre de mots en dessous duquel un abstract est considéré "court".
    is_short_abstract=True pour ces articles = signal de qualité faible.
    """

    enabled: bool = True
    """
    False → toutes les métriques sont à 0 / False / '' sans calcul.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.short_abstract_threshold < 1:
            raise ValueError(
                f"'short_abstract_threshold' doit être >= 1 "
                f"(reçu : {self.short_abstract_threshold})"
            )
