"""
Détection de la langue principale des articles arXiv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..config.lang_settings import LangConfig

logger = logging.getLogger(__name__)


# Résultat de détection
@dataclass(frozen=True)
class LangDetectionResult:
    """
    Résultat de la détection de langue pour une partition.
    """

    langs: list[str]
    confidences: list[float]

    def __len__(self) -> int:
        return len(self.langs)

    def to_dataframe(self, index: Optional[pd.Index] = None) -> pd.DataFrame:
        """
        Convertit le résultat en DataFrame avec colonnes 'lang' et 'lang_confidence'.
        """
        return pd.DataFrame(
            {
                "lang": pd.Series(self.langs, dtype="string"),
                "lang_confidence": pd.array(self.confidences, dtype="float32"),
            },
            index=index,
        )


# LangDetector
class LangDetector:
    """
    Détecte la langue principale de chaque article avec score de confiance.
    """

    def __init__(self, config: Optional[LangConfig] = None) -> None:
        self._cfg = config or LangConfig()
        self._backend_loaded: bool = False
        self._detect_fn = None

    # API publique
    def detect(self, texts: list[str]) -> LangDetectionResult:
        """
        Détecte la langue de chaque texte avec son score de confiance.
        """
        self._ensure_backend()

        if not texts:
            return LangDetectionResult(langs=[], confidences=[])

        logger.debug(
            "Détection de langue : %d textes | backend=%s",
            len(texts),
            self._cfg.backend,
        )

        langs: list[str] = []
        confidences: list[float] = []

        for text in texts:
            lang, conf = self._detect_single(text)
            langs.append(lang)
            confidences.append(conf)

        n_fallback = sum(1 for c in confidences if c == self._cfg.fallback_confidence)
        if n_fallback > len(texts) * 0.3:
            logger.warning(
                "Taux de fallback élevé : %d/%d textes (confidence=%.1f). "
                "Vérifier la longueur des textes (min=%d chars) ou changer de backend.",
                n_fallback,
                len(texts),
                self._cfg.fallback_confidence,
                self._cfg.min_text_length,
            )

        return LangDetectionResult(langs=langs, confidences=confidences)

    # Méthodes internes
    def _detect_single(self, text: str) -> tuple[str, float]:
        """
        Détecte la langue d'un texte individuel.
        """
        if not text or len(text) < self._cfg.min_text_length:
            return self._cfg.fallback_lang, self._cfg.fallback_confidence

        try:
            return self._detect_fn(text)
        except Exception as exc:
            logger.debug(
                "Échec détection langue (backend=%s) : %s",
                self._cfg.backend,
                exc,
            )
            return self._cfg.fallback_lang, self._cfg.fallback_confidence

    def _ensure_backend(self) -> None:
        """
        Charge le backend de détection de langue si ce n'est pas encore fait.

        Raises:
            ImportError      : si la librairie n'est pas installée.
            FileNotFoundError: si le modèle fasttext est introuvable.
        """
        if self._backend_loaded:
            return

        backend = self._cfg.backend
        logger.info("Initialisation backend langue : '%s'", backend)

        if backend == "fasttext":
            self._detect_fn = self._make_fasttext_fn()
        else:
            raise ValueError(f"Backend inconnu : '{backend}'")

        self._backend_loaded = True
        logger.info("Backend '%s' prêt.", backend)

    def _make_fasttext_fn(self):
        """
        Construit le callable de détection pour le backend fasttext.

        Charge le modèle depuis config.fasttext_model_path.
        """
        try:
            import fasttext
        except ImportError as exc:
            raise ImportError("Le backend 'fasttext' requiert fasttext.") from exc

        import os

        model_path = self._cfg.fasttext_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modèle fasttext introuvable : '{model_path}'. "
                f"Télécharger lid.176.bin ou lid.176.ftz depuis "
                f"https://fasttext.cc/docs/en/language-identification.html"
            )

        ft_model = fasttext.load_model(model_path)

        def _detect(text: str) -> tuple[str, float]:
            # fasttext préfère les textes sur une seule ligne
            clean = text.replace("\n", " ").strip()
            labels, probs = ft_model.predict(clean, k=1)
            if labels:
                # labels[0] = '__label__en' → strip prefix
                lang_code = labels[0].replace("__label__", "")
                return lang_code, round(float(probs[0]), 4)
            return self._cfg.fallback_lang, self._cfg.fallback_confidence

        return _detect
