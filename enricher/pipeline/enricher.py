"""
Orchestrateur principal du pipeline d'enrichissement global. 

Phase 1 : Entraînement BERTopic global par catégorie
Phase 2 : Enrichissement NLP partition par partition + drift check
Phase 3 : Snapshot + History (non-bloquant)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config.settings import EnricherConfig
from ..monitoring.report import EnrichmentReport, PartitionStats
from ..pipeline.gold_schema import validate_gold_dataframe
from ..storage.base_storage import BaseEnricherStorage, CleanPartition
from ..trend_detection.snapshot.partition_snapshot import TopicSnapshot
from ..trend_detection.storage.trend_storage import (
    BaseTrendStorage,
    build_trend_storage,
)
from .nlp_pipeline import NlpPipeline
from .topic_trainer import TopicTrainer
from ..topic_modeling.drift_detector import DriftDetector


class ArxivEnricher:
    """Orchestre l'enrichissement NLP de la zone CLEAN : GOLD."""

    def __init__(
        self,
        config: EnricherConfig,
        storage: Optional[BaseEnricherStorage] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._cfg = config
        self.logger = logger or logging.getLogger(__name__)

        Path(config.models_path).mkdir(parents=True, exist_ok=True)

        self._storage = storage or _build_default_storage(config)
        self._pipeline = NlpPipeline(config, log=self.logger)
        self._trainer = TopicTrainer(self._pipeline, self._storage, log=self.logger)
        self._snapshot = TopicSnapshot(config.trend)
        self._trend_storage = _build_trend_storage(config)
        self._drift_detector = DriftDetector(config.topic)

    # API publique
    def run(self) -> EnrichmentReport:
        """Lance l'enrichissement complet de toutes les partitions CLEAN disponibles."""
        report = EnrichmentReport()

        partitions = self._storage.discover(
            categories=self._cfg.categories,
            years=self._cfg.years,
            months=self._cfg.months,
        )

        if not partitions:
            self.logger.warning("Aucune partition CLEAN trouvée. Vérifiez clean_path.")
            report.finalize()
            return report

        self.logger.info("%d partitions découvertes.", len(partitions))

        # Phase 1 : entraînement BERTopic si nécessaire
        self._trainer.fit_if_needed(partitions)

        # Phase 2+3 : enrichissement partition par partition
        for partition in partitions:
            stats = self._enrich_one_partition(partition, partitions)
            report.add_partition(stats)

        report.finalize()
        report.log_summary(self.logger)
        return report

    # Traitement d'une partition
    def _enrich_one_partition(
        self,
        partition: CleanPartition,
        all_partitions: list[CleanPartition],
    ) -> PartitionStats:
        """Enrichit une partition CLEAN et écrit GOLD + TRENDS."""
        stats = PartitionStats(
            category=partition.category,
            year=partition.year,
            month=partition.month,
        )

        self.logger.info(
            "Partition : %s/%d/%02d",
            partition.category,
            partition.year,
            partition.month,
        )

        if self._cfg.skip_existing and self._storage.gold_exists(partition):
            if not self._storage.clean_is_newer_than_gold(partition):
                self.logger.info("  GOLD existant : ignoré.")
                stats.n_skipped = 1
                return stats
            self.logger.info("  CLEAN plus récent que GOLD : re-enrichissement.")

        try:
            df = self._storage.read_clean(partition)
        except Exception as exc:
            self.logger.error("  Lecture CLEAN échouée : %s", exc)
            stats.error = str(exc)
            return stats

        if df.empty:
            self.logger.warning("  DataFrame CLEAN vide : ignoré.")
            stats.n_articles = 0
            return stats

        n = len(df)
        stats.n_articles = n

        # Phase 2 : Pipeline NLP
        try:
            df_gold = self._pipeline.run(df, partition.category, partition)
        except Exception as exc:
            self.logger.error("  Pipeline NLP échoué : %s", exc)
            stats.n_failed = n
            stats.error = str(exc)
            return stats

        # Drift check : re-fit si taux d'outliers > seuil
        # En cas d'échec du re-fit, on conserve le df_gold du premier run (non-bloquant)
        if "topic_id" in df_gold.columns:
            outlier_rate = float((df_gold["topic_id"] == -1).mean())
            if self._drift_detector.should_refit(outlier_rate):
                self.logger.warning(
                    "  Drift détecté (outlier_rate=%.1f%%) > seuil %.1f%% : re-fit.",
                    100 * outlier_rate,
                    100 * self._drift_detector.threshold,
                )
                try:
                    mapping = self._trainer.refit_window(
                        partition.category, all_partitions
                    )
                    self._trend_storage.rebuild_history(partition.category)
                    df_gold = self._pipeline.run(df, partition.category, partition)

                    # Remapping des topic_ids pour maintenir la continuité des séries
                    # temporelles après re-fit (BERTopic renumérote les topics).
                    if mapping and "topic_id" in df_gold.columns:
                        df_gold["topic_id"] = df_gold["topic_id"].apply(
                            lambda tid: (
                                mapping.get(int(tid), int(tid))
                                if int(tid) != -1
                                else -1
                            )
                        )
                        self.logger.info(
                            "  TopicAligner : %d topic_ids remappés.", len(mapping)
                        )
                except Exception as exc:
                    self.logger.warning("  Re-fit échoué (non-bloquant) : %s", exc)

        for err in validate_gold_dataframe(df_gold):
            self.logger.warning("  Validation GOLD : %s", err)

        _fill_partition_stats(stats, df_gold, n)

        # Écriture GOLD
        try:
            self._storage.write_gold(df_gold, partition)
            stats.n_enriched = n
            self.logger.info("  GOLD écrit (%d articles).", n)
        except Exception as exc:
            self.logger.error("  Écriture GOLD échouée : %s", exc)
            stats.n_failed = n
            stats.error = str(exc)
            return stats

        # Phase 3 : Snapshot + History
        try:
            snapshot = self._snapshot.compute(df_gold)
            if len(snapshot) > 0:
                month = partition.month_key
                category = partition.category
                snapshot_df = snapshot.to_history_dataframe(month)
                self._trend_storage.write_snapshot(snapshot_df, category, month)
                self._trend_storage.update_history(snapshot_df, category, month)
        except Exception as exc:
            self.logger.warning("  Snapshot/History échoué (non-bloquant) : %s", exc)

        return stats


# Helpers
def _build_trend_storage(config: EnricherConfig) -> BaseTrendStorage:
    return build_trend_storage(config.storage_backend, config.trends_path)


def _build_default_storage(config: EnricherConfig) -> BaseEnricherStorage:
    if config.storage_backend == "minio":
        from ..storage.minio_storage import MinioEnricherStorage

        return MinioEnricherStorage()
    from ..storage.local_storage import LocalEnricherStorage

    return LocalEnricherStorage(config)


def _fill_partition_stats(
    stats: PartitionStats,
    df_gold: pd.DataFrame,
    n: int,
) -> None:
    """Remplit les statistiques depuis le DataFrame GOLD enrichi."""
    if "topic_id" in df_gold.columns:
        n_outliers = int((df_gold["topic_id"] == -1).sum())
        stats.n_topic_outliers = n_outliers
        stats.topic_outlier_pct = 100.0 * n_outliers / n if n > 0 else 0.0
        stats.n_topics = int(
            df_gold.loc[df_gold["topic_id"] != -1, "topic_id"].nunique()
        )

    if "lang" in df_gold.columns:
        n_non_en = int((df_gold["lang"] != "en").sum())
        stats.n_non_english = n_non_en
        stats.non_english_pct = 100.0 * n_non_en / n if n > 0 else 0.0

    if "is_short_abstract" in df_gold.columns:
        stats.n_short_abstract = int(df_gold["is_short_abstract"].sum())
    if "novelty_score" in df_gold.columns:
        stats.avg_novelty_score = float(df_gold["novelty_score"].mean())
    if "intra_similarity" in df_gold.columns:
        stats.avg_intra_similarity = float(df_gold["intra_similarity"].mean())
    if "entity_list" in df_gold.columns:
        avg_ents = df_gold["entity_list"].apply(len).mean()
        stats.avg_entities_per_article = (
            float(avg_ents) if not np.isnan(avg_ents) else 0.0
        )
