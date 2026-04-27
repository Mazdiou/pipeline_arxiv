"""
Configuration du prétraitement NLP des textes CLEAN.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """
    Paramètres de NlpTextPreprocessor.
    """

    include_abstract: bool = True
    """
    Inclure l'abstract dans le texte NLP ou se limiter au titre.

    True  : texte NLP = title + abstract
    False : texte NLP = title seul
    """

    min_cleaned_length: int = 20
    """
    Longueur minimale (en caractères) du texte nettoyé.

    Si le texte nettoyé est plus court (ex: abstract entièrement LaTeX),
    NlpTextPreprocessor bascule sur le titre seul pour conserver
    un signal sémantique minimal.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.min_cleaned_length < 0:
            raise ValueError(
                f"'min_cleaned_length' doit être >= 0 (reçu : {self.min_cleaned_length})"
            )
