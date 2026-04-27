"""
TopicSnapshot : agregation statistique des topics d'une partition GOLD.

Produit un SnapshotResult avec 14 metriques par topic :
  Identite       : topic_id, topic_label
  Volume         : n_articles, pct_articles
  Nouveaute      : mean_novelty
  Mots-cles      : top_keywords
  Coherence      : mean_centroid_distance, mean_topic_prob
  Lisibilite     : mean_flesch_grade
  Langue         : dominant_lang, pct_english, mean_lang_confidence
  Interdiscipl.  : pct_cross_category
  LaTeX          : mean_latex_density

Colonnes GOLD consommees :
  Obligatoire    : topic_id
  Optionnelles   : topic_label, novelty_score, keywords, centroid_distance,
                   topic_prob, flesch_kincaid_grade, lang, lang_confidence,
                   categories, latex_density
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import pandas as pd

from ...config.trend_settings import TrendConfig
from .snapshot_result import SnapshotResult

logger = logging.getLogger(__name__)


class TopicSnapshot:
    """
    Calcule un snapshot statistique par topic depuis un DataFrame GOLD.

    Args:
        config : TrendConfig avec min_articles_per_topic et top_keywords_per_topic.
    """

    def __init__(self, config: TrendConfig) -> None:
        self._cfg = config

    def compute(self, df: pd.DataFrame) -> SnapshotResult:
        """
        Agregation par topic_id sur un DataFrame GOLD.

        Args:
            df : DataFrame GOLD avec colonnes NLP (topic_id obligatoire).

        Returns:
            SnapshotResult avec une ligne par topic (hors outliers topic_id=-1).
            Vide si df vide ou sans colonne topic_id.
        """
        if df.empty or "topic_id" not in df.columns:
            return SnapshotResult()

        # Exclure les outliers BERTopic (topic_id == -1)
        df_valid = df[df["topic_id"] != -1]
        if df_valid.empty:
            return SnapshotResult()

        n_total = len(df)
        keyword_texts = _explode_keywords(df_valid)

        topic_ids: list[int] = []
        topic_labels: list[str] = []
        n_articles: list[int] = []
        pct_articles: list[float] = []
        mean_novelty: list[float] = []
        top_keywords: list[list[str]] = []
        mean_centroid_distance: list[float] = []
        mean_topic_prob: list[float] = []
        mean_flesch_grade: list[float] = []
        dominant_lang: list[str] = []
        pct_english: list[float] = []
        mean_lang_confidence: list[float] = []
        pct_cross_category: list[float] = []
        mean_latex_density: list[float] = []

        for tid, group in df_valid.groupby("topic_id"):
            n = len(group)
            if n < self._cfg.min_articles_per_topic:
                continue

            tid_int = int(tid)
            topic_ids.append(tid_int)
            n_articles.append(n)
            pct_articles.append(round(100.0 * n / n_total, 2))

            # Topic label
            if "topic_label" in group.columns:
                mode_vals = group["topic_label"].mode()
                topic_labels.append(
                    str(mode_vals.iloc[0]) if len(mode_vals) > 0 else f"topic_{tid}"
                )
            else:
                topic_labels.append(f"topic_{tid}")

            # Métriques numériques simples
            mean_novelty.append(_safe_mean(group, "novelty_score"))
            mean_centroid_distance.append(_safe_mean(group, "centroid_distance"))
            mean_topic_prob.append(_safe_mean(group, "topic_prob"))
            mean_flesch_grade.append(_safe_mean(group, "flesch_kincaid_grade"))
            mean_latex_density.append(_safe_mean(group, "latex_density"))
            mean_lang_confidence.append(_safe_mean(group, "lang_confidence"))

            # Top keywords
            kw_group = keyword_texts.loc[
                keyword_texts["topic_id"] == tid, "keyword_text"
            ].dropna()
            counter = Counter(kw_group)
            top_keywords.append(
                [kw for kw, _ in counter.most_common(self._cfg.top_keywords_per_topic)]
            )

            # Langue
            dom, pct_en = _compute_lang_stats(group)
            dominant_lang.append(dom)
            pct_english.append(pct_en)

            # Interdisciplinarité
            pct_cross_category.append(_compute_pct_cross_category(group))

        return SnapshotResult(
            topic_id=topic_ids,
            topic_label=topic_labels,
            n_articles=n_articles,
            pct_articles=pct_articles,
            mean_novelty=mean_novelty,
            top_keywords=top_keywords,
            mean_centroid_distance=mean_centroid_distance,
            mean_topic_prob=mean_topic_prob,
            mean_flesch_grade=mean_flesch_grade,
            dominant_lang=dominant_lang,
            pct_english=pct_english,
            mean_lang_confidence=mean_lang_confidence,
            pct_cross_category=pct_cross_category,
            mean_latex_density=mean_latex_density,
        )


# Helpers
def _extract_keyword_text(item) -> Optional[str]:
    """
    Extrait le texte d'un mot-cle depuis un tuple ou un dict.

    Formats supportes :
      - tuple[str, float]                     → format memoire
      - dict{"keyword": str, "score": float}  → format Parquet
    """
    if isinstance(item, dict):
        return item.get("keyword")
    if isinstance(item, (tuple, list)) and len(item) >= 1:
        return str(item[0])
    return None


def _explode_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explose la colonne 'keywords' et extrait les textes.

    Retourne un DataFrame avec colonnes ['topic_id', 'keyword_text'].
    """
    if "keywords" not in df.columns:
        return pd.DataFrame(columns=["topic_id", "keyword_text"])

    exploded = df[["topic_id", "keywords"]].explode("keywords")
    exploded["keyword_text"] = exploded["keywords"].apply(_extract_keyword_text)
    return exploded[["topic_id", "keyword_text"]]


def _safe_mean(group: pd.DataFrame, col: str) -> float:
    """Calcule la moyenne d'une colonne avec fallback 0.0 si absente ou vide."""
    if col not in group.columns:
        return 0.0
    val = pd.to_numeric(group[col], errors="coerce").mean()
    return round(float(val), 4) if pd.notna(val) else 0.0


def _compute_lang_stats(group: pd.DataFrame) -> tuple[str, float]:
    """
    Calcule la langue dominante et le pourcentage d'articles anglais.

    Returns:
        (dominant_lang, pct_english) — fallback ("en", 100.0) si colonne absente.
    """
    if "lang" not in group.columns:
        return "en", 100.0

    lang_series = group["lang"].dropna()
    if len(lang_series) == 0:
        return "en", 100.0

    mode_vals = lang_series.mode()
    dom = str(mode_vals.iloc[0]) if len(mode_vals) > 0 else "en"
    pct_en = round(100.0 * (lang_series == "en").sum() / len(lang_series), 2)
    return dom, float(pct_en)


def _compute_pct_cross_category(group: pd.DataFrame) -> float:
    """
    Calcule le pourcentage d'articles couvrant plusieurs domaines arXiv.

    Un article est cross-category si ses catégories span plus d'un préfixe
    principal (ex: "cs.AI" + "stat.ML" → domaines "cs" et "stat" → True).

    Returns:
        Pourcentage [0.0, 100.0] — fallback 0.0 si colonne absente.
    """
    if "categories" not in group.columns:
        return 0.0

    n_cross = sum(
        1 for cats in group["categories"].dropna() if _is_cross_category(cats)
    )
    return round(100.0 * n_cross / len(group), 2)


def _is_cross_category(categories) -> bool:
    """
    True si l'article couvre plusieurs domaines arXiv principaux.

    Domaine principal = préfixe avant le premier '.'
    Exemples : "cs.AI" → "cs", "stat.ML" → "stat", "quant-ph" → "quant-ph"
    """
    try:
        cats_list = list(categories)
    except (TypeError, ValueError):
        return False

    domains: set[str] = set()
    for cat in cats_list:
        if isinstance(cat, str) and cat:
            domains.add(cat.split(".")[0])
    return len(domains) > 1
