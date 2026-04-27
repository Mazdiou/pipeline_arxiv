"""
Backend filesystem local pour le processeur arXiv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base_storage import BaseProcessorStorage, RawPartition
from ..io.reader import discover_raw_files, read_raw_file
from ..io.writer import write_clean_file


class LocalProcessorStorage(BaseProcessorStorage):
    """
    Stockage processeur sur le filesystem local.
    """

    def __init__(
        self,
        raw_path: str = "data/RAW",
        clean_path: str = "data/CLEAN",
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise le stockage processeur sur le filesystem local.
        """
        self.raw_root = Path(raw_path)
        self.clean_root = Path(clean_path)
        self.logger = logger or logging.getLogger("arxiv.processor.storage")

    # Interface publique
    def discover(
        self,
        category: str = "",
        year: int | None = None,
        month: int | None = None,
    ) -> list[RawPartition]:
        """
        Découvre les partitions RAW locales correspondant aux filtres.
        """
        paths = discover_raw_files(str(self.raw_root), category, year, month)
        partitions: list[RawPartition] = []
        for path in paths:
            parts = path.parts
            try:
                partitions.append(
                    RawPartition(
                        category=parts[-4],
                        year=int(parts[-3]),
                        month=int(parts[-2]),
                    )
                )
            except (IndexError, ValueError):
                self.logger.warning(f"Chemin RAW inattendu ignoré : {path}")
        return sorted(partitions, key=lambda p: (p.category, p.year, p.month))

    def read_raw(self, partition: RawPartition) -> pd.DataFrame:
        """
        Lit les articles RAW locaux d'une partition.
        """
        path = (
            self.raw_root
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.json"
        )
        self.logger.debug(f"Lecture RAW local : {path}")
        return read_raw_file(path)

    def clean_exists(self, partition: RawPartition) -> bool:
        """Vérifie si le fichier CLEAN Parquet local existe pour cette partition."""
        path = (
            self.clean_root
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.parquet"
        )
        return path.exists()

    def raw_is_newer_than_clean(self, partition: RawPartition) -> bool:
        """Retourne True si le fichier RAW est plus récent que le fichier CLEAN."""
        raw_path = (
            self.raw_root
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.json"
        )
        clean_path = (
            self.clean_root
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.parquet"
        )
        try:
            return raw_path.stat().st_mtime > clean_path.stat().st_mtime
        except OSError:
            return False

    def write_clean(self, df: pd.DataFrame, partition: RawPartition) -> str:
        """
        Écrit le DataFrame nettoyé en Parquet local.
        """
        out_path = write_clean_file(
            df,
            clean_path=str(self.clean_root),
            category=partition.category,
            year=partition.year,
            month=partition.month,
        )
        self.logger.debug(f"CLEAN écrit : {out_path}")
        return str(out_path)
