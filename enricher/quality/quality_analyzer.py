"""
Analyse de la qualité NLP et extraction de métriques textuelles des articles.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ..config.quality_settings import QualityConfig

logger = logging.getLogger(__name__)

# Patterns de segmentation et d'analyse textuelle
_RE_SENTENCES = re.compile(r"[.!?]+\s+")
_RE_VOWEL_GROUPS = re.compile(r"[aeiouyAEIOUY]+")
_RE_WORDS = re.compile(r"\b[a-zA-Z']+\b")


@dataclass(frozen=True)
class QualityResult:
    """
    Résultat de l'analyse de qualité pour une partition d'articles.
    """

    flesch_kincaid_grade: list
    is_short_abstract: list
    latex_density: list

    def __len__(self) -> int:
        return len(self.flesch_kincaid_grade)

    def to_dataframe(self, index: Optional[Any] = None) -> pd.DataFrame:
        """
        Convertit le résultat en DataFrame avec 3 colonnes typées.
        """
        return pd.DataFrame(
            {
                "flesch_kincaid_grade": pd.array(
                    self.flesch_kincaid_grade, dtype="float32"
                ),
                "is_short_abstract": pd.array(self.is_short_abstract, dtype="bool"),
                "latex_density": pd.array(self.latex_density, dtype="float32"),
            },
            index=index,
        )


class ArticleQualityAnalyzer:
    """
    Analyse la qualité textuelle et extrait des métriques NLP des articles arXiv.
    """

    def __init__(self, config: Optional[QualityConfig] = None) -> None:
        """
        Args:
            config: Configuration de l'analyseur.
        """
        self._cfg = config or QualityConfig()

    def analyze(
        self,
        df: pd.DataFrame,
        latex_densities: Optional[list[float]] = None,
    ) -> QualityResult:
        """
        Calcule toutes les métriques de qualité pour les articles du DataFrame.
        """
        n = len(df)

        if not self._cfg.enabled:
            logger.debug("QualityAnalyzer désactivé.")
            return QualityResult(
                flesch_kincaid_grade=[0.0] * n,
                is_short_abstract=[False] * n,
                latex_density=[0.0] * n,
            )

        abstracts = df["abstract"].fillna("").astype(str).tolist()

        word_counts = [_count_words(t) for t in abstracts]
        flesch_kincaid = [_flesch_kincaid_grade(t) for t in abstracts]
        is_short = [wc < self._cfg.short_abstract_threshold for wc in word_counts]
        ld = latex_densities if latex_densities is not None else [0.0] * n

        logger.debug(
            "QualityAnalyzer : n=%d | short_abstracts=%d | avg_words=%.1f",
            n,
            sum(is_short),
            sum(word_counts) / n if n > 0 else 0.0,
        )

        return QualityResult(
            flesch_kincaid_grade=flesch_kincaid,
            is_short_abstract=is_short,
            latex_density=ld,
        )


# Fonctions de calcul (niveau article)
def _count_words(text: str) -> int:
    """Compte les mots alphabétiques d'un texte (exclut chiffres, ponctuation)."""
    return len(_RE_WORDS.findall(text)) if text else 0


def _count_sentences(text: str) -> int:
    """Compte les phrases d'un texte en utilisant une heuristique simple (.!?)."""
    sentences = _RE_SENTENCES.split(text.strip())
    return len([s for s in sentences if s])


def _count_syllables(word: str) -> int:
    """
    Compte le nombre approximatif de syllabes dans un mot anglais.
    """
    word = word.lower().rstrip("e")
    syllables = len(_RE_VOWEL_GROUPS.findall(word))
    return max(1, syllables)


def _flesch_kincaid_grade(text: str) -> float:
    """
    Calcule le niveau Flesch-Kincaid Grade Level (années d'études US).

    Formule : 0.39 × (mots/phrases) + 11.8 × (syllabes/mots) - 15.59
    Score clampé à [0, 25].
    """
    words = _RE_WORDS.findall(text)
    n_words = len(words)
    if n_words == 0:
        return 0.0

    n_sentences = _count_sentences(text)
    n_syllables = sum(_count_syllables(w) for w in words)

    grade = (
        0.39 * (n_words / max(1, n_sentences)) + 11.8 * (n_syllables / n_words) - 15.59
    )
    return float(max(0.0, min(25.0, grade)))
