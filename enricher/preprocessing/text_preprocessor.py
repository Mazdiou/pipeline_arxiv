"""
Prétraitement NLP des textes CLEAN pour les modèles sémantiques.

Pipeline NLP (prepare_texts) :
    1. Suppression des placeholders [MATH] (laissés par processor/)
    2. Suppression des URLs résiduelles (http/https)
    3. Suppression des identifiants arXiv (arXiv:2302.12345)
    4. Suppression des citations numériques [1], [1,2,3], [12–15]
    5. Suppression des citations parenthétiques (Smith et al., 2023)
    6. Suppression des caractères spéciaux non-textuels
    7. Normalisation des espaces
    8. Fallback sur le titre si le texte nettoyé est trop court

Pipeline langue (prepare_lang_texts) :
    1. Suppression des placeholders [MATH]
    2. Suppression des URLs et identifiants arXiv
    3. Normalisation des espaces
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from ..config.preprocessing_settings import PreprocessingConfig


# Patterns compilés au chargement du module (une seule fois)

# Placeholder [MATH] généré par processor/core/cleaner.py
# Représente toute expression mathématique LaTeX ($...$, $$...$$, \begin{...})
_RE_MATH_PLACEHOLDER = re.compile(r"\[MATH\]")

# URLs : http / https avec chemins, query strings et fragments
_RE_URLS = re.compile(r"https?://\S+")

# Identifiants arXiv : arXiv:2302.12345 ou arXiv:2302.12345v2
# \b : word boundary pour ne pas matcher "arXiv" au milieu d'un mot
_RE_ARXIV_ID = re.compile(r"\barxiv:\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)

# Citations numériques entre crochets : [1], [1,2], [12–15], [1; 3]
# \u2013 = en-dash, \u2014 = em-dash (utilisés dans les plages de numéros)
_RE_CITE_BRACKETS = re.compile(r"\[\s*\d[\d,;\s\u2013\u2014\-]{0,20}\]")

# Citations parenthétiques : (Smith et al., 2023), (Doe 2020), (LeCun, 2015)
# Heuristique : parenthèse contenant une année 4 chiffres (19xx ou 20xx)
_RE_CITE_PARENS = re.compile(
    r"\([^)]{2,80}(?:19|20)\d{2}[^)]{0,30}\)",
    re.IGNORECASE,
)

# Caractères spéciaux non-textuels pour NLP
# Conserve : lettres unicode (\w), espaces (\s), tirets (-), apostrophes ('), points (.)
# Supprime : symboles mathématiques, ponctuation lourde, caractères de contrôle
_RE_SPECIAL_NLP = re.compile(r"[^\w\s\-'.]", re.UNICODE)

# Espaces multiples — appliqué après toutes les substitutions
_RE_SPACES = re.compile(r"\s+")


class NlpTextPreprocessor:
    """
    Prépare les textes CLEAN pour les modèles NLP sémantiques.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self._cfg = config or PreprocessingConfig()

    # API publique
    def prepare_texts(self, df: pd.DataFrame) -> list[str]:
        """
        Construit et nettoie les textes pour l'analyse sémantique NLP.
        """
        titles = df["title"].fillna("").astype(str)
        abstracts = (
            df["abstract"].fillna("").astype(str)
            if self._cfg.include_abstract
            else [""] * len(df)
        )

        return [
            self._prepare_nlp_single(title, abstract)
            for title, abstract in zip(titles, abstracts)
        ]

    def prepare_lang_texts(self, df: pd.DataFrame) -> list[str]:
        """
        Construit les textes pour la détection de langue (nettoyage léger).
        """
        titles = df["title"].fillna("").astype(str)
        abstracts = df["abstract"].fillna("").astype(str)

        return [
            self._prepare_lang_single(title, abstract)
            for title, abstract in zip(titles, abstracts)
        ]

    def compute_latex_density(self, df: pd.DataFrame) -> list[float]:
        """
        Calcule la densité LaTeX de chaque article.
        """
        titles = df["title"].fillna("").astype(str)
        abstracts = df["abstract"].fillna("").astype(str)

        result: list[float] = []
        for title, abstract in zip(titles, abstracts):
            text = f"{title} {abstract}"
            n_math = len(_RE_MATH_PLACEHOLDER.findall(text))
            # Mots hors placeholders [MATH]
            text_no_math = _RE_MATH_PLACEHOLDER.sub(" ", text)
            n_words = len(text_no_math.split())
            total = n_math + n_words
            result.append(round(n_math / total, 4) if total > 0 else 0.0)
        return result

    # Méthodes internes
    def _prepare_nlp_single(self, title: str, abstract: str) -> str:
        """Pipeline NLP complet sur un article (titre + abstract)."""
        raw = f"{title} {abstract}".strip()
        cleaned = _clean_for_nlp(raw)

        if len(cleaned) < self._cfg.min_cleaned_length:
            # Fallback : le texte nettoyé est trop court (ex: abstract entièrement
            # LaTeX → tous les [MATH] supprimés). On retourne le titre seul,
            # éventuellement nettoyé, pour conserver un signal sémantique minimal.
            cleaned = _clean_for_nlp(title) or title.strip()

        return cleaned

    @staticmethod
    def _prepare_lang_single(title: str, abstract: str) -> str:
        """Nettoyage léger pour la détection de langue (un article)."""
        raw = f"{title} {abstract}".strip()
        return _clean_for_lang(raw)


# Fonctions pures de nettoyage (stateless — testables indépendamment)
def _clean_for_nlp(text: str) -> str:
    """
    Pipeline de nettoyage complet pour l'analyse sémantique NLP.
    """
    text = _RE_MATH_PLACEHOLDER.sub(" ", text)
    text = _RE_URLS.sub(" ", text)
    text = _RE_ARXIV_ID.sub(" ", text)
    text = _RE_CITE_BRACKETS.sub(" ", text)
    text = _RE_CITE_PARENS.sub(" ", text)
    text = _RE_SPECIAL_NLP.sub(" ", text)
    return _RE_SPACES.sub(" ", text).strip()


def _clean_for_lang(text: str) -> str:
    """
    Pipeline de nettoyage léger pour la détection de langue.

    Supprime les artefacts clairement non-linguistiques sans toucher
    à la ponctuation ni aux caractères spéciaux du texte naturel.
    """
    text = _RE_MATH_PLACEHOLDER.sub(" ", text)
    text = _RE_URLS.sub(" ", text)
    text = _RE_ARXIV_ID.sub(" ", text)
    return _RE_SPACES.sub(" ", text).strip()
