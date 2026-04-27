"""
Configuration de la détection de langue des articles arXiv.

fasttext    Modèle Facebook LID (Language Identification).
            176 langues, ~0.1 ms/texte, très rapide et précis.
            Nécessite le fichier modèle lid.176.ftz
            
            Installation : pip install fasttext
            Modèle     : https://fasttext.cc/docs/en/language-identification.html
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LangConfig:
    """
    Paramètres du module LangDetector.
    """

    backend: str = "fasttext"
    """
    Moteur de détection de langue.
    """

    fasttext_model_path: str = "models/lid.176.ftz"
    """
    Chemin vers le fichier modèle fasttext (.bin ou .ftz).
    """

    min_text_length: int = 30
    """
    Longueur minimale du texte (en caractères) pour tenter la détection.
    """

    fallback_lang: str = "en"
    """
    Code ISO 639-1 retourné quand la détection échoue ou est impossible.
    """

    fallback_confidence: float = 0.0
    """
    Score de confiance retourné avec fallback_lang.
    0.0 indique explicitement que la détection n'a pas abouti.
    """

    seed: int = 0
    """
    Graine aléatoire pour la reproductibilité de langdetect.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:

        if self.min_text_length < 0:
            raise ValueError(
                f"'min_text_length' doit être >= 0 (reçu : {self.min_text_length})"
            )

        if not self.fallback_lang or len(self.fallback_lang) != 2:
            raise ValueError(
                f"'fallback_lang' doit être un code ISO 639-1 à 2 lettres "
                f"(reçu : '{self.fallback_lang}')"
            )

        if not (0.0 <= self.fallback_confidence <= 1.0):
            raise ValueError(
                f"'fallback_confidence' doit être dans [0.0, 1.0] "
                f"(reçu : {self.fallback_confidence})"
            )
