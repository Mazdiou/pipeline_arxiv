"""
Contrat abstrait du stockage pour le module enricher.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class CleanPartition:
    """
    Identifie une partition CLEAN par ses 3 dimensions.
    """

    category: str
    """Categorie arXiv normalisee (ex: 'cs_AI', 'stat_ML')."""

    year: int
    """Annee de publication (ex: 2026)."""

    month: int
    """Mois de publication 1-12 (ex: 2)."""

    def __str__(self) -> str:
        return f"{self.category}/{self.year}/{self.month:02d}"

    @property
    def month_key(self) -> str:
        """Cle temporelle au format YYYY-MM."""
        return f"{self.year:04d}-{self.month:02d}"


class BaseEnricherStorage(ABC):
    """
    Interface abstraite pour le stockage CLEAN -> GOLD.
    """

    @abstractmethod
    def discover(
        self,
        categories: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        months: Optional[List[int]] = None,
    ) -> List[CleanPartition]:
        """
        Liste toutes les partitions CLEAN disponibles.

        Args:
            categories: Filtre par liste de categories (ex: ['cs_AI', 'stat_ML']).
                        Si None, toutes les categories sont incluses.
            years     : Filtre par liste d'annees (ex: [2020, 2021]). Si None, toutes les annees sont incluses.
            months    : Filtre par liste de mois (1-12). Si None, tous les mois sont inclus.

        Returns:
            Liste de partitions trouvees, triees par (categorie, annee, mois).
        """

    @abstractmethod
    def read_clean(self, partition: CleanPartition) -> pd.DataFrame:
        """
        Lit le DataFrame CLEAN d'une partition.

        Args:
            partition: Partition a lire.

        Returns:
            DataFrame avec toutes les colonnes du schema CLEAN.

        Raises:
            FileNotFoundError: Si la partition n'existe pas.
        """

    @abstractmethod
    def gold_exists(self, partition: CleanPartition) -> bool:
        """
        Verifie si un fichier GOLD existe deja pour cette partition.

        Args:
            partition: Partition a verifier.

        Returns:
            True si le fichier GOLD existe.
        """

    @abstractmethod
    def clean_is_newer_than_gold(self, partition: CleanPartition) -> bool:
        """
        Retourne True si le fichier CLEAN a ete modifie apres le fichier GOLD.

        Args:
            partition : Partition a verifier.

        Returns:
            True si CLEAN plus recent que GOLD -> re-enrichissement necessaire.
        """
        ...

    @abstractmethod
    def write_gold(self, df: pd.DataFrame, partition: CleanPartition) -> None:
        """
        Ecrit le DataFrame GOLD enrichi pour une partition.

        Args:
            df       : DataFrame GOLD avec toutes les colonnes NLP.
            partition: Partition cible.

        Raises:
            Exception: En cas d'erreur d'ecriture.
        """
