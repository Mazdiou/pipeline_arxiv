"""
Construit des séries temporelles complètes par topic_id à partir du snapshot courant et de l'historique cumulatif.
Chaque série est un TopicTimeSeries, aligné sur un axe temporel commun (mois) et rempli de valeurs neutres pour les mois d'absence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..snapshot.snapshot_result import SnapshotResult

logger = logging.getLogger(__name__)

# Valeurs neutres pour les mois où un topic est absent.
# pct_english vaut 100.0 par convention (pas 0.0) : un topic sans données
# n'est pas "non anglophone", il est simplement non observé ce mois-là.
_NEUTRAL_FLOAT = 0.0
_NEUTRAL_LANG = "en"
_NEUTRAL_PCT_EN = 100.0


@dataclass(frozen=True)
class TopicTimeSeries:
    """
    Série temporelle complète d'un topic sur N mois.

    Toutes les listes sont alignées sur `months` (même longueur, même ordre).
    Un mois sans articles reçoit des valeurs neutres : jamais de trou dans la série.

    Métriques de volume :
        n_articles            : Nombre d'articles publiés ce mois.
        pct_articles          : Part du topic dans la catégorie (%).

    Métriques sémantiques (héritées de l'enricher) :
        mean_novelty          : Isolement sémantique moyen : les articles du topic
                                sont-ils proches ou éloignés des autres clusters ?
                                Élevé = topic encore marginal, non absorbé.
        mean_centroid_distance: Dispersion intra-topic (cohésion géométrique).
                                À ne pas confondre avec semantic_drift qui mesure
                                le MOUVEMENT du centroïde dans le temps.
        mean_topic_prob       : Confiance d'assignation BERTopic ∈ [0, 1].
                                Faible = articles à la frontière entre topics.

    Métriques qualitatives :
        mean_flesch_grade     : Complexité textuelle (proxy de maturité du topic).
        mean_latex_density    : Densité de formules (proxy de technicité).
        pct_cross_category    : % d'articles couvrant plusieurs domaines arXiv.
        pct_english           : % d'articles en anglais (portée internationale).
        mean_lang_confidence  : Qualité de détection de langue.
        dominant_lang_by_month: Langue dominante par mois.

    Mots-clés :
        top_keywords_by_month : Top keywords KeyBERT par mois : utilisés par
                                compute_keyword_drift() pour mesurer la dérive lexicale.
    """

    topic_id: int = -1
    months: list[str] = field(default_factory=list)

    # Volume
    n_articles: list[int] = field(default_factory=list)
    pct_articles: list[float] = field(default_factory=list)

    # Sémantique
    mean_novelty: list[float] = field(default_factory=list)
    mean_centroid_distance: list[float] = field(default_factory=list)
    mean_topic_prob: list[float] = field(default_factory=list)

    # Qualité
    mean_flesch_grade: list[float] = field(default_factory=list)
    mean_latex_density: list[float] = field(default_factory=list)
    pct_cross_category: list[float] = field(default_factory=list)
    pct_english: list[float] = field(default_factory=list)
    mean_lang_confidence: list[float] = field(default_factory=list)
    dominant_lang_by_month: list[str] = field(default_factory=list)

    # Mots-clés
    top_keywords_by_month: list[list[str]] = field(default_factory=list)

    # Propriétés dérivées
    @property
    def n_months_active(self) -> int:
        """Nombre de mois où le topic a au moins 1 article."""
        return sum(1 for n in self.n_articles if n > 0)

    @property
    def first_seen(self) -> Optional[str]:
        """Premier mois où le topic apparaît. None si jamais actif."""
        for i, n in enumerate(self.n_articles):
            if n > 0:
                return self.months[i]
        return None

    @property
    def last_seen(self) -> Optional[str]:
        """Dernier mois où le topic apparaît. None si jamais actif."""
        for i in range(len(self.n_articles) - 1, -1, -1):
            if self.n_articles[i] > 0:
                return self.months[i]
        return None

    @property
    def volume_series(self) -> list[float]:
        """
        Série de volume en float, avec zéros pour les mois d'absence.

        Utilisée par z-score, burst et CAGR (série complète).
        Pour Mann-Kendall et Hurst, filtrer les zéros avec :
            vol_active = [v for v in volume_series if v > 0]
        """
        return [float(n) for n in self.n_articles]


class TimeSeriesBuilder:
    """
    Construit un TopicTimeSeries par topic_id depuis l'historique cumulatif.

    Algorithme :
        1. Fusionner le snapshot courant avec l'historique passé.
        2. Déduire l'axe temporel commun (union des mois observés).
        3. Pour chaque topic, construire sa série en remplissant les mois
           manquants par des valeurs neutres.

    L'index MultiIndex (month, topic_id) réduit chaque lookup à O(1)
    et évite de filtrer le DataFrame complet à chaque itération.
    """

    def build(
        self,
        current_snapshot: SnapshotResult,
        current_month: str,
        history_df: pd.DataFrame,
    ) -> dict[int, TopicTimeSeries]:
        """
        Construit un TopicTimeSeries par topic_id.

        Args:
            current_snapshot : SnapshotResult de la partition courante.
            current_month    : Clé temporelle "YYYY-MM".
            history_df       : DataFrame history cumulatif (peut être vide).

        Returns:
            dict[topic_id] → TopicTimeSeries, vide si aucune donnée disponible.
        """
        df_current = current_snapshot.to_history_dataframe(current_month)

        if history_df.empty:
            combined = df_current
        else:
            # Exclure le mois courant de l'historique pour éviter les doublons
            # si le pipeline est rejoué sur le même mois.
            past = history_df[history_df["month"] != current_month]
            combined = pd.concat([past, df_current], ignore_index=True)

        if combined.empty:
            return {}

        all_months = sorted(combined["month"].unique())
        all_topic_ids = sorted(combined["topic_id"].dropna().astype(int).unique())

        # MultiIndex (month, topic_id) → accès O(1) par (mois, topic).
        indexed = combined.set_index(["month", "topic_id"])

        result: dict[int, TopicTimeSeries] = {}
        for tid in all_topic_ids:
            result[tid] = self._build_single(tid, all_months, indexed)

        logger.debug("TimeSeries : %d topics sur %d mois", len(result), len(all_months))
        return result

    def _build_single(
        self,
        tid: int,
        all_months: list[str],
        indexed: pd.DataFrame,
    ) -> TopicTimeSeries:
        """
        Construit la série temporelle d'un seul topic.

        Pour chaque mois de l'axe temporel commun, tente de lire la ligne
        (month, topic_id) dans le MultiIndex. Si absente (KeyError), remplit
        avec les valeurs neutres appropriées.
        """
        months: list[str] = []
        n_articles: list[int] = []
        pct_articles: list[float] = []
        mean_novelty: list[float] = []
        mean_centroid_distance: list[float] = []
        mean_topic_prob: list[float] = []
        mean_flesch_grade: list[float] = []
        mean_latex_density: list[float] = []
        pct_cross_category: list[float] = []
        pct_english: list[float] = []
        mean_lang_confidence: list[float] = []
        dominant_lang_by_month: list[str] = []
        top_keywords_by_month: list[list[str]] = []

        for month_key in all_months:
            months.append(month_key)
            try:
                row = indexed.loc[(month_key, tid)]
                n_articles.append(int(row["n_articles"]))
                pct_articles.append(float(row["pct_articles"]))
                mean_novelty.append(_safe_float(row.get("mean_novelty")))
                mean_centroid_distance.append(
                    _safe_float(row.get("mean_centroid_distance"))
                )
                mean_topic_prob.append(_safe_float(row.get("mean_topic_prob")))
                mean_flesch_grade.append(_safe_float(row.get("mean_flesch_grade")))
                mean_latex_density.append(_safe_float(row.get("mean_latex_density")))
                pct_cross_category.append(_safe_float(row.get("pct_cross_category")))
                pct_english.append(_safe_float(row.get("pct_english"), _NEUTRAL_PCT_EN))
                mean_lang_confidence.append(
                    _safe_float(row.get("mean_lang_confidence"))
                )
                dominant_lang_by_month.append(
                    _safe_str(row.get("dominant_lang"), _NEUTRAL_LANG)
                )
                top_keywords_by_month.append(
                    _normalize_keywords(row.get("top_keywords"))
                )
            except KeyError:
                # Le topic n'avait pas d'articles ce mois : valeurs neutres.
                n_articles.append(0)
                pct_articles.append(_NEUTRAL_FLOAT)
                mean_novelty.append(_NEUTRAL_FLOAT)
                mean_centroid_distance.append(_NEUTRAL_FLOAT)
                mean_topic_prob.append(_NEUTRAL_FLOAT)
                mean_flesch_grade.append(_NEUTRAL_FLOAT)
                mean_latex_density.append(_NEUTRAL_FLOAT)
                pct_cross_category.append(_NEUTRAL_FLOAT)
                pct_english.append(_NEUTRAL_PCT_EN)
                mean_lang_confidence.append(_NEUTRAL_FLOAT)
                dominant_lang_by_month.append(_NEUTRAL_LANG)
                top_keywords_by_month.append([])

        return TopicTimeSeries(
            topic_id=tid,
            months=months,
            n_articles=n_articles,
            pct_articles=pct_articles,
            mean_novelty=mean_novelty,
            mean_centroid_distance=mean_centroid_distance,
            mean_topic_prob=mean_topic_prob,
            mean_flesch_grade=mean_flesch_grade,
            mean_latex_density=mean_latex_density,
            pct_cross_category=pct_cross_category,
            pct_english=pct_english,
            mean_lang_confidence=mean_lang_confidence,
            dominant_lang_by_month=dominant_lang_by_month,
            top_keywords_by_month=top_keywords_by_month,
        )


# Helpers
def _safe_float(val: object, default: float = _NEUTRAL_FLOAT) -> float:
    """
    Convertit val en float de manière défensive.

    Retourne `default` si val est None, NaN, ou non convertible.
    Nécessaire car les colonnes optionnelles lues depuis Parquet peuvent
    contenir des NaN pandas ou des None Python selon leur provenance.
    """
    try:
        f = float(val)  # type: ignore[arg-type]
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _safe_str(val: object, default: str) -> str:
    """
    Convertit val en str de manière défensive.

    Retourne `default` si val est None ou un NaN flottant.
    Protège contre les NaN pandas qui s'infiltrent dans les colonnes string
    lors de la lecture Parquet sur des mois sans données.
    """
    if val is None:
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
    except TypeError:
        pass
    return str(val)


def _normalize_keywords(raw: object) -> list[str]:
    """
    Normalise les keywords lus depuis Parquet en list[str].

    Parquet peut stocker les listes comme numpy arrays (via PyArrow) ou comme
    listes Python selon le schéma d'écriture. Les deux cas sont gérés.
    """
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        # numpy array ou pandas array
        return [str(x) for x in raw.tolist()]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return []
