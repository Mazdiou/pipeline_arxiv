"""
TopicClassifier : classification d'un topic en 6 états selon ses métriques.

Ordre de priorité (du plus au moins prioritaire) :

    NEW          : topic trop récent pour être évalué (< 2 mois d'historique)
    EMERGING     : volume faible, mais croissance rapide et articles novateurs
    TRANSFORMING : dérive sémantique forte sans variation de volume notable
    TRENDING     : croissance soutenue, persistante et statistiquement significative
    DECLINING    : déclin confirmé par les tests statistiques
    STABLE       : aucune tendance détectée (état par défaut)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TopicStatus(str, Enum):
    """État d'un topic tel que déterminé par le module de tendance."""

    NEW = "NEW"
    """Historique insuffisant : le topic est apparu récemment."""

    EMERGING = "EMERGING"
    """Signal faible : volume encore marginal, mais croissance rapide."""

    TRANSFORMING = "TRANSFORMING"
    """Mutation sémantique : le contenu évolue sans que le volume ne varie."""

    TRENDING = "TRENDING"
    """Croissance soutenue, confirmée statistiquement (Mann-Kendall + Hurst)."""

    DECLINING = "DECLINING"
    """Recul du volume, confirmé statistiquement."""

    STABLE = "STABLE"
    """Aucune tendance notable détectée. État attribué par défaut."""


@dataclass(frozen=True)
class ClassifierThresholds:
    """
    Seuils de classification, regroupés pour faciliter les expériences comparatives.

    Attributs :

    new_max_months_active : int
        Nombre maximal de mois actifs pour être classé NEW.
        En dessous de ce seuil, les métriques sont jugées non fiables.

    emerging_max_articles : int
        Volume absolu maximal (dernier mois) pour le critère EMERGING.
        Complété par ``emerging_max_pct_articles`` pour les petites catégories.

    emerging_max_pct_articles : float
        Part maximale du topic dans sa catégorie (%) au dernier mois,
        utilisée comme seuil EMERGING relatif. Évite de classer EMERGING
        un topic qui domine une petite catégorie mais dépasse le seuil absolu.

    emerging_min_emergence : float
        Score d'émergence minimal pour déclencher EMERGING.

    trending_min_mk_tau : float
        Tau de Mann-Kendall minimal pour TRENDING (direction de la tendance).

    trending_max_mk_pvalue : float
        P-value maximale pour que la tendance TRENDING soit retenue.
        Filtre les faux positifs sur des séries courtes où tau peut être
        élevé par hasard.

    trending_min_hurst : float
        Exposant de Hurst minimal pour confirmer la persistance de la tendance.
        Requis uniquement si la série comporte au moins
        ``trending_min_months_for_hurst`` mois actifs.

    trending_min_months_for_hurst : int
        Longueur minimale de la série (mois actifs) pour que l'exposant de Hurst
        soit exigé. En dessous, Hurst est peu fiable et Mann-Kendall suffit.

    declining_max_mk_tau : float
        Tau de Mann-Kendall maximal (valeur négative) pour DECLINING.

    declining_max_mk_pvalue : float
        P-value maximale pour DECLINING. Seuil délibérément plus souple que
        TRENDING, car le déclin est moins actionnable mais reste utile pour
        filtrer le bruit de fond.

    transforming_min_drift : float
        Dérive sémantique minimale (distance cosinus) pour TRANSFORMING.
        Calibré sur les embeddings SPECTER/all-MiniLM : les dérives mensuelles
        observées sur stat.ML s'échelonnent dans [0.02, 0.15]. Un seuil de 0.10
        capture les mutations sémantiques notables tout en évitant le bruit.

    transforming_max_mk_abs : float
        Valeur absolue maximale de tau pour TRANSFORMING. Garantit l'absence
        de tendance volumétrique nette (ni croissance, ni déclin).
    """

    new_max_months_active: int = 2
    emerging_max_articles: int = 30
    emerging_max_pct_articles: float = 5.0
    emerging_min_emergence: float = 0.15
    trending_min_mk_tau: float = 0.40
    trending_max_mk_pvalue: float = 0.05
    trending_min_hurst: float = 0.55
    trending_min_months_for_hurst: int = 10
    declining_max_mk_tau: float = -0.30
    declining_max_mk_pvalue: float = 0.10
    transforming_min_drift: float = 0.10
    transforming_max_mk_abs: float = 0.25


class TopicClassifier:
    """
    Classifie un topic selon 6 états mutuellement exclusifs, par règles de priorité.

    L'ordre d'évaluation est fixe et reflète une hiérarchie métier :
    les états les plus informatifs ou les moins bien couverts par les approches
    volumétriques classiques sont traités en premier.

        1. NEW          : données insuffisantes : aucune autre règle ne s'applique
        2. EMERGING     : signal faible prioritaire (intérêt académique fort)
        3. TRANSFORMING : mutation sémantique invisible aux approches volumétriques
        4. TRENDING     : croissance confirmée
        5. DECLINING    : déclin confirmé
        6. STABLE       : état résiduel, attribué si aucune règle ne se déclenche
    """

    def __init__(
        self, thresholds: ClassifierThresholds = ClassifierThresholds()
    ) -> None:
        self._t = thresholds

    def classify(
        self,
        n_months_active: int,
        n_articles_last: int,
        mann_kendall_tau: float,
        mann_kendall_pvalue: float,
        hurst_exponent: float,
        emergence_score_last: float,
        semantic_drift_last: float,
        pct_articles_last: float = 100.0,
    ) -> TopicStatus:
        """
        Détermine l'état d'un topic à partir de ses métriques courantes.

        Paramètres :

        n_months_active : int
            Nombre de mois où le topic a publié au moins un article.
        n_articles_last : int
            Volume du dernier mois connu.
        mann_kendall_tau : float
            Tau de Mann-Kendall calculé sur la série temporelle de volume
            (positif = tendance croissante, négatif = décroissante).
        mann_kendall_pvalue : float
            P-value associée au test Mann-Kendall.
        hurst_exponent : float
            Exposant de Hurst de la série de volume
            (> 0.5 indique une persistance de la tendance).
        emergence_score_last : float
            Score d'émergence du dernier mois (combinaison volume + nouveauté).
        semantic_drift_last : float
            Distance cosinus entre l'embedding moyen du dernier mois et celui
            du mois précédent. Passer 0.0 si le module DST est désactivé.
        pct_articles_last : float, optional
            Part du topic dans sa catégorie arXiv (%) au dernier mois.
            Utilisé comme critère EMERGING relatif (défaut : 100.0).
        """
        t = self._t

        # 1. NEW
        # Historique trop court : les métriques statistiques ne sont pas fiables.
        if n_months_active <= t.new_max_months_active:
            return TopicStatus.NEW

        # 2. EMERGING
        # Signal faible au sens d'Ansoff : activité anormalement forte
        # dans un domaine encore marginal.
        # Le critère de volume est évalué en OR absolu/relatif pour couvrir
        # les petites catégories où 30 articles peut représenter un topic dominant.
        is_low_volume = (
            n_articles_last <= t.emerging_max_articles
            or pct_articles_last <= t.emerging_max_pct_articles
        )
        if is_low_volume and emergence_score_last >= t.emerging_min_emergence:
            return TopicStatus.EMERGING

        # 3. TRANSFORMING
        # Mutation sémantique sans variation de volume notable.
        # Ce cas est invisible aux approches purement volumétriques : tau faible
        # (pas de tendance) mais dérive cosinus élevée (changement de contenu).
        if (
            semantic_drift_last >= t.transforming_min_drift
            and abs(mann_kendall_tau) <= t.transforming_max_mk_abs
        ):
            return TopicStatus.TRANSFORMING

        # 4. TRENDING
        # Croissance soutenue et persistante, confirmée par deux tests :
        # - Mann-Kendall valide la direction et la significativité statistique.
        # - Hurst confirme la persistance (auto-corrélation positive de la série).
        #   Si la série est trop courte (< trending_min_months_for_hurst),
        #   Hurst est peu fiable et n'est pas exigé.
        hurst_ok = (
            n_months_active < t.trending_min_months_for_hurst
            or hurst_exponent >= t.trending_min_hurst
        )
        mk_significant = mann_kendall_pvalue <= t.trending_max_mk_pvalue
        if mann_kendall_tau >= t.trending_min_mk_tau and hurst_ok and mk_significant:
            return TopicStatus.TRENDING

        # 5. DECLINING
        # Tendance baissière confirmée. Seuil de p-value plus souple que TRENDING
        # car le déclin est moins actionnable, mais reste utile pour filtrer le bruit.
        if (
            mann_kendall_tau <= t.declining_max_mk_tau
            and mann_kendall_pvalue <= t.declining_max_mk_pvalue
        ):
            return TopicStatus.DECLINING

        # 6. STABLE
        return TopicStatus.STABLE
