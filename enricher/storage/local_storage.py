"""
Implementation locale du storage enricher (systeme de fichiers).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config.settings import EnricherConfig
from ..pipeline.gold_schema import (
    GOLD_SCHEMA,
    GOLD_SCHEMA_COLUMNS,
    prepare_gold_dataframe,
)
from .base_storage import BaseEnricherStorage, CleanPartition

logger = logging.getLogger(__name__)


class LocalEnricherStorage(BaseEnricherStorage):
    """
    Storage local : lit CLEAN depuis le disque, écrit GOLD sur le disque.
    """

    def __init__(self, config: EnricherConfig) -> None:
        self._clean_path = Path(config.clean_path)
        self._gold_path = Path(config.gold_path)

    def discover(
        self,
        categories: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        months: Optional[List[int]] = None,
    ) -> List[CleanPartition]:
        """
        Découvre les partitions CLEAN disponibles dans le filesystem local.

        Structure attendue : {clean_path}/{category}/{YYYY}/{MM}/articles.parquet
        """
        partitions: List[CleanPartition] = []

        if not self._clean_path.exists():
            logger.warning("Chemin CLEAN introuvable : %s", self._clean_path)
            return partitions

        for cat_dir in sorted(self._clean_path.iterdir()):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            if categories and category not in categories:
                continue

            for year_dir in sorted(cat_dir.iterdir()):
                if not year_dir.is_dir():
                    continue
                try:
                    year = int(year_dir.name)
                except ValueError:
                    continue
                if years and year not in years:
                    continue

                for month_dir in sorted(year_dir.iterdir()):
                    if not month_dir.is_dir():
                        continue
                    try:
                        month = int(month_dir.name)
                    except ValueError:
                        continue
                    if months and month not in months:
                        continue

                    parquet_file = month_dir / "articles.parquet"
                    if parquet_file.exists():
                        partitions.append(
                            CleanPartition(category=category, year=year, month=month)
                        )

        logger.debug("%d partitions CLEAN découvertes.", len(partitions))
        return partitions

    def read_clean(self, partition: CleanPartition) -> pd.DataFrame:
        """Lit le Parquet CLEAN depuis le disque."""
        path = (
            self._clean_path
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"Partition CLEAN introuvable : {path}")
        return pd.read_parquet(path)

    def gold_exists(self, partition: CleanPartition) -> bool:
        """Vérifie si le fichier GOLD existe déjà."""
        path = (
            self._gold_path
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.parquet"
        )
        return path.exists()

    def clean_is_newer_than_gold(self, partition: CleanPartition) -> bool:
        """Retourne True si le fichier CLEAN est plus récent que le fichier GOLD."""
        clean_path = (
            self._clean_path
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.parquet"
        )
        gold_path = (
            self._gold_path
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.parquet"
        )
        try:
            return clean_path.stat().st_mtime > gold_path.stat().st_mtime
        except OSError:
            return False

    def write_gold(self, df: pd.DataFrame, partition: CleanPartition) -> None:
        """Écrit le DataFrame GOLD sur le disque (atomic write via fichier tmp)."""
        out_path = (
            self._gold_path
            / partition.category
            / f"{partition.year:04d}"
            / f"{partition.month:02d}"
            / "articles.parquet"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df_prepared = prepare_gold_dataframe(df)
        cols_present = [c for c in GOLD_SCHEMA_COLUMNS if c in df_prepared.columns]
        df_ordered = df_prepared[cols_present]
        schema_fields = [f for f in GOLD_SCHEMA if f.name in cols_present]
        partial_schema = pa.schema(schema_fields)

        table = pa.Table.from_pandas(
            df_ordered,
            schema=partial_schema,
            preserve_index=False,
            safe=False,
        )
        _write_atomic(table, out_path)
        logger.info("GOLD écrit : %s (%d articles)", out_path, len(df))


# Helper interne
def _write_atomic(table: pa.Table, out_path: Path) -> None:
    """Écriture atomique Parquet via fichier .tmp.parquet puis rename."""
    tmp_path = out_path.with_suffix(".tmp.parquet")
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
