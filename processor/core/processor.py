"""Orchestrateur principal du pipeline de traitement arXiv RAW to CLEAN."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..config.settings import ProcessorConfig
from ..storage.base_storage import BaseProcessorStorage, RawPartition
from ..storage.local_storage import LocalProcessorStorage
from .cleaner import clean_text
from .validator import validate_dataframe
from .report import ProcessorReport


class ArxivProcessor:
    """Orchestrateur du pipeline RAW to CLEAN."""

    def __init__(
        self,
        config: ProcessorConfig,
        logger: logging.Logger | None = None,
        storage: BaseProcessorStorage | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("arxiv.processor")

        self._storage = storage or LocalProcessorStorage(
            raw_path=config.raw_data_path,
            clean_path=config.clean_data_path,
            logger=self.logger,
        )

    def run(self) -> ProcessorReport:
        """Point d'entrée principal. Traite toutes les partitions RAW éligibles
        et retourne un rapport consolidé (même en cas d'erreurs partielles)."""
        start = datetime.now(timezone.utc)
        self._log_start()

        # discover() filtre selon category/year/month de la config (None = tout)
        partitions = self._storage.discover(
            category=self.config.category,
            year=self.config.year,
            month=self.config.month,
        )

        if not partitions:
            return self._empty_report(start)

        metrics = {
            "raw": 0,
            "clean": 0,
            "dropped_invalid": 0,
            "dropped_duplicate": 0,
            "files": [],
            "errors": [],
        }

        for partition in partitions:
            try:
                result = self._process_one_partition(partition)
                if not result:
                    continue

                metrics["raw"] += result["raw"]
                metrics["clean"] += result["clean"]
                metrics["dropped_invalid"] += result["dropped_invalid"]
                metrics["dropped_duplicate"] += result["dropped_duplicate"]

                if result["clean_path"]:
                    metrics["files"].append(result["clean_path"])

            except Exception as exc:
                # Erreur isolée à la partition : on continue les autres
                msg = f"{partition} : {exc}"
                self.logger.exception(f"ERREUR : {msg}")
                metrics["errors"].append(msg)

        return self._build_report(start, metrics)

    # Traitement d'une partition
    def _process_one_partition(self, partition: RawPartition) -> dict | None:
        if not self._should_process(partition):
            return None

        self.logger.info(f"Traitement : {partition}")

        df = self._storage.read_raw(partition)
        if df.empty:
            self.logger.warning(f"Partition vide : {partition}")
            return None

        raw_count = len(df)

        # Validation : écarte les lignes sans champs obligatoires (arxiv_id, title…)
        valid_df, dropped = validate_dataframe(df)
        n_invalid = len(dropped)

        # Déduplication sur arxiv_id_short (identifiant canonique sans numéro de version)
        before = len(valid_df)
        valid_df = valid_df.drop_duplicates(subset=["arxiv_id_short"])
        n_dup = before - len(valid_df)

        if valid_df.empty:
            return self._empty_partition_result(raw_count, n_invalid, n_dup)

        # Nettoyage
        self._clean_text_columns(valid_df)

        # Enrichissement
        self._add_metadata(valid_df, partition)

        # Écriture
        path = self._storage.write_clean(valid_df, partition)

        self.logger.info(
            f"CLEAN {path} | {raw_count} → {len(valid_df)} "
            f"(−{n_invalid} invalides, −{n_dup} doublons)"
        )

        return {
            "raw": raw_count,
            "clean": len(valid_df),
            "dropped_invalid": n_invalid,
            "dropped_duplicate": n_dup,
            "clean_path": path,
        }

    # Helpers
    def _should_process(self, partition: RawPartition) -> bool:
        """Décide si une partition doit être (re)traitée.

        Ordre de priorité :
        1. Pas encore de fichier CLEAN : toujours traiter
        2. overwrite_existing=True dans la config : forcer le retraitement
        3. Fichier RAW plus récent que CLEAN : retraitement automatique
        """
        if not self._storage.clean_exists(partition):
            return True

        if self.config.overwrite_existing:
            return True

        if self._storage.raw_is_newer_than_clean(partition):
            self.logger.info(f"Retraitement : {partition}")
            return True

        self.logger.info(f"Ignoré : {partition}")
        return False

    def _clean_text_columns(self, df):
        for col in ("title", "abstract", "comment", "journal_ref"):
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

    def _add_metadata(self, df, partition):
        # Colonnes dérivées des auteurs
        df["authors_count"] = df["authors"].apply(len)
        df["first_author"] = df["authors"].apply(lambda a: a[0] if a else None)

        # Colonnes de traçabilité
        df["_processed_at"] = datetime.now(timezone.utc).isoformat()
        df["_partition_category"] = partition.category
        df["_partition_year"] = partition.year
        df["_partition_month"] = partition.month

    def _empty_partition_result(self, raw, invalid, dup):
        return {
            "raw": raw,
            "clean": 0,
            "dropped_invalid": invalid,
            "dropped_duplicate": dup,
            "clean_path": None,
        }

    def _empty_report(self, start):
        return ProcessorReport.build(
            start=start,
            end=datetime.now(timezone.utc),
            files_processed=[],
            total_raw=0,
            total_clean=0,
            total_dropped_invalid=0,
            total_dropped_duplicate=0,
            errors=[],
            config_snapshot=self.config.to_dict(),
        )

    def _build_report(self, start, m):
        end = datetime.now(timezone.utc)

        report = ProcessorReport.build(
            start=start,
            end=end,
            files_processed=m["files"],
            total_raw=m["raw"],
            total_clean=m["clean"],
            total_dropped_invalid=m["dropped_invalid"],
            total_dropped_duplicate=m["dropped_duplicate"],
            errors=m["errors"],
            config_snapshot=self.config.to_dict(),
        )

        for line in report.summary_lines():
            self.logger.info(line)

        return report

    def _log_start(self):
        self.logger.info("=" * 60)
        self.logger.info("START arXiv processor")
        self.logger.info(self.config.summary())
        self.logger.info(f"Storage: {type(self._storage).__name__}")
        self.logger.info("=" * 60)
