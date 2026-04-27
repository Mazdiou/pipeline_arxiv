"""
Configuration du module d'extraction de mots-clés KeyBERT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KeywordConfig:
    """
    Paramètres du module ArticleKeywordExtractor.
    """

    # KeyBERT
    top_n: int = 7
    """
    Nombre de mots-clés extraits par article.
    """

    keyphrase_ngram_range: tuple[int, int] = field(default_factory=lambda: (1, 2))
    """
    Plage de n-grammes pour les keyphrases.
    """

    stop_words: Optional[str] = "english"
    """
    Stop words à exclure des keyphrases.
    """

    # MMR
    use_mmr: bool = True
    """
    True : Maximal Marginal Relevance pour diversifier les mots-clés.
    False : plus rapide mais moins diversifié.
    """

    diversity: float = 0.55
    """
    Paramètre de diversité MMR [0.0, 1.0].
    0.0 : mots-clés très similaires entre eux (redondants).
    1.0 : mots-clés maximalement diversifiés (peut perdre en pertinence).
    Ignoré si use_mmr=False.
    """

    # CountVectorizer
    min_df: int = 2
    """
    Fréquence minimale d'un terme dans le corpus pour être candidat keyphrase.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.top_n < 1:
            raise ValueError(f"'top_n' doit être >= 1 (reçu : {self.top_n})")
        if not (0.0 <= self.diversity <= 1.0):
            raise ValueError(
                f"'diversity' doit être dans [0.0, 1.0] (reçu : {self.diversity})"
            )
        if self.min_df < 1:
            raise ValueError(f"'min_df' doit être >= 1 (reçu : {self.min_df})")
        if (
            len(self.keyphrase_ngram_range) != 2
            or self.keyphrase_ngram_range[0] < 1
            or self.keyphrase_ngram_range[1] < self.keyphrase_ngram_range[0]
        ):
            raise ValueError(
                f"'keyphrase_ngram_range' invalide : {self.keyphrase_ngram_range}. "
                f"Attendu : tuple (min, max) avec 1 <= min <= max."
            )
