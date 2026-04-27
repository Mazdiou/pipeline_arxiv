"""
Interface abstraite pour les backends de stockage du processeur.

Le processeur effectue deux opérations de stockage :
    1. Lecture des fichiers RAW JSON produits par le collector
    2. Écriture des fichiers CLEAN Parquet après traitement

Pour brancher un nouveau backend (GCS, Azure Blob...) :
    Créer une classe héritant de BaseProcessorStorage
    et implémenter les cinq méthodes abstraites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RawPartition:
    """
    Identifie une partition RAW : catégorie / année / mois.
    """

    category: str
    year: int
    month: int

    def __str__(self) -> str:
        return f"{self.category}/{self.year:04d}/{self.month:02d}"


class BaseProcessorStorage(ABC):
    """
    Interface commune pour les backends de stockage du processeur arXiv.
    """

    @abstractmethod
    def discover(
        self,
        category: str = "",
        year: int | None = None,
        month: int | None = None,
    ) -> list[RawPartition]:
        """
        Découvre les partitions RAW disponibles correspondant aux filtres.
        """
        ...

    @abstractmethod
    def read_raw(self, partition: RawPartition) -> pd.DataFrame:
        """
        Lit les articles RAW d'une partition.
        """
        ...

    @abstractmethod
    def clean_exists(self, partition: RawPartition) -> bool:
        """
        Vérifie si le fichier CLEAN Parquet existe pour cette partition.
        """
        ...

    @abstractmethod
    def raw_is_newer_than_clean(self, partition: RawPartition) -> bool:
        """
        Retourne True si le fichier RAW a été modifié après le fichier CLEAN.
        """
        ...

    @abstractmethod
    def write_clean(self, df: pd.DataFrame, partition: RawPartition) -> str:
        """
        Écrit le DataFrame nettoyé en Parquet dans la zone CLEAN.
        """
        ...
