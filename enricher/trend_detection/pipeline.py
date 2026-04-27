"""
TrendPipeline : orchestrateur batch mensuel du module trend_detection.

Flux d'exécution pour une catégorie et un mois donnés :

    1. [Snapshot]     TopicSnapshot.compute(gold_df)
                      → SnapshotResult (14 métriques agrégées par topic)

    2. [Storage]      TrendStorage.write_snapshot() + update_history()
                      → Persistance des snapshots bruts et de l'historique

    3. [Ingestion]    HistoryReader.read() + TimeSeriesBuilder.build()
                      → dict[topic_id] → TopicTimeSeries

    4. [DST]          EmbeddingAggregator.compute_centroids()  (si activé)
                      → dict[topic_id] → dict[mois] → centroïde

    5. [Analyse]      ApproachHybrid.analyze() pour chaque topic
                      → dict[topic_id] → HybridResult

    6. [Output]       TrendFormatter.build_report() → to_json() + to_parquet()
                      → Fichiers de sortie pour le dashboard
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..config.trend_settings import TrendConfig
from .classification.signal_scorer import ScorerWeights
from .classification.topic_classifier import ClassifierThresholds
from .comparison.approach_hybrid import ApproachHybrid, HybridResult
from .core.time_series_builder import TimeSeriesBuilder, TopicTimeSeries
from .ingestion.embedding_aggregator import EmbeddingAggregator
from .ingestion.history_reader import HistoryReader
from .output.formatter import CategoryTrendReport, TrendFormatter
from .snapshot.partition_snapshot import TopicSnapshot
from .snapshot.snapshot_result import SnapshotResult
from .storage.trend_storage import BaseTrendStorage

logger = logging.getLogger(__name__)


class TrendPipeline:
    """
    Orchestrateur batch mensuel du module trend_detection.

    Exécute les 6 étapes du pipeline dans l'ordre et retourne un rapport
    prêt à être consommé par le dashboard Streamlit.

    Workflow
    --------
    1. Snapshot   : calcul des métriques agrégées par topic à partir du GOLD.
    2. Storage    : persistance du snapshot brut et mise à jour de l'historique.
    3. Ingestion  : construction des séries temporelles (historique + mois courant).
    4. DST        : (optionnel) agrégation des centroïdes d'embedding par topic/mois.
    5. Analyse    : classification hybride (état, score, burst, etc.) pour chaque topic.
    6. Output     : formattage et export JSON/Parquet pour le dashboard.

    Parameters
    ----------
    config : TrendConfig
        Configuration générale du module (seuils, fenêtres, flags d'activation…).
    storage : BaseTrendStorage
        Instance de stockage (local ou MinIO) pour les snapshots et l'historique.
    weights : ScorerWeights, optional
        Pondérations α, β, γ pour le score composite (approche D).
    thresholds : ClassifierThresholds, optional
        Seuils de classification des 6 états (TRENDING, EMERGING, etc.).
    """

    def __init__(
        self,
        config: TrendConfig,
        storage: BaseTrendStorage,
        weights: ScorerWeights = ScorerWeights(),
        thresholds: ClassifierThresholds = ClassifierThresholds(),
    ) -> None:
        self._cfg = config
        self._storage = storage

        # Étape 1 : Snapshot
        self._snapshotter = TopicSnapshot(config)

        # Étape 3 : Construction des séries temporelles
        self._ts_builder = TimeSeriesBuilder()
        self._history_reader = HistoryReader(storage)

        # Étape 4 : Agrégation des embeddings (DST)
        self._emb_aggregator: Optional[EmbeddingAggregator] = None
        if config.semantic_drift_enabled:
            self._emb_aggregator = EmbeddingAggregator(
                gold_path=config.gold_data_path,
                embeddings_cache_path=config.embeddings_cache_path,
                model_name=config.embeddings_model_name,
            )

        # Étape 5 : Analyse hybride
        self._hybrid = ApproachHybrid(
            z_score_window=config.z_score_window,
            hurst_min_len=config.hurst_min_length,
            burst_max_states=config.burst_max_states,
            weights=weights,
            thresholds=thresholds,
        )

        # Étape 6 : Formattage
        self._formatter = TrendFormatter()

    def run(
        self,
        gold_df: pd.DataFrame,
        category: str,
        month: str,
        output_json_path: Optional[str] = None,
        output_parquet_path: Optional[str] = None,
    ) -> Optional[CategoryTrendReport]:
        """
        Exécute le pipeline complet pour une catégorie et un mois.

        Parameters
        ----------
        gold_df : pd.DataFrame
            DataFrame GOLD du mois courant (topics, métriques, embeddings…).
        category : str
            Catégorie arXiv (ex: "cs_AI", "stat_ML").
        month : str
            Mois de calcul au format "YYYY-MM".
        output_json_path : Optional[str], default=None
            Chemin de sortie pour le rapport JSON (optionnel).
        output_parquet_path : Optional[str], default=None
            Chemin de sortie pour le rapport Parquet (optionnel).

        Returns
        -------
        Optional[CategoryTrendReport]
            Rapport complet si le pipeline s'exécute normalement,
            None si le module est désactivé ou si les données sont insuffisantes.
        """
        # Vérification des prérequis
        if not self._cfg.enabled:
            logger.info("TrendPipeline désactivé (config.enabled=False).")
            return None

        if gold_df.empty:
            logger.warning(
                "TrendPipeline : DataFrame GOLD vide pour %s/%s.", category, month
            )
            return None

        logger.info("TrendPipeline démarré : %s / %s", category, month)

        # Étape 1 : Snapshot
        snapshot = self._snapshotter.compute(gold_df)
        if len(snapshot) == 0:
            logger.warning(
                "Snapshot vide pour %s/%s : aucun topic valide.", category, month
            )
            return None
        logger.debug("Snapshot : %d topics", len(snapshot))

        # Étape 2 : Persistance
        snapshot_df = snapshot.to_history_dataframe(month)
        self._storage.write_snapshot(snapshot_df, category, month)
        self._storage.update_history(snapshot_df, category, month)

        # Étape 3 : Séries temporelles
        # to_month garantit qu'on ne voit que l'historique disponible
        # au moment du run : évite la fuite de données entre mois
        history_df = self._history_reader.read(category, to_month=month)
        time_series_map: dict[int, TopicTimeSeries] = self._ts_builder.build(
            current_snapshot=snapshot,
            current_month=month,
            history_df=history_df,
        )
        logger.debug("Séries temporelles : %d topics", len(time_series_map))

        # Filtrage : seuls les topics avec assez d'historique sont éligibles
        eligible = {
            tid: ts
            for tid, ts in time_series_map.items()
            if ts.n_months_active >= self._cfg.min_months_for_trend
        }
        logger.debug(
            "Topics éligibles (≥%d mois actifs) : %d / %d",
            self._cfg.min_months_for_trend,
            len(eligible),
            len(time_series_map),
        )

        # Étape 4 : Centroïdes DST (si activé)
        all_months = sorted({m for ts in time_series_map.values() for m in ts.months})
        centroids_by_topic: dict[int, dict[str, "np.ndarray"]] = {}

        if self._emb_aggregator and eligible:
            centroids_by_topic = self._emb_aggregator.compute_centroids(
                category=category,
                months=all_months,
            )
            logger.debug(
                "DST : centroids calculés pour %d topics", len(centroids_by_topic)
            )

        # Étape 5 : Analyse hybride
        hybrid_results: dict[int, HybridResult] = {}
        for tid, ts in eligible.items():
            centroids = centroids_by_topic.get(tid)
            hybrid_results[tid] = self._hybrid.analyze(ts, centroids)

        logger.debug("Analyse hybride : %d topics analysés", len(hybrid_results))

        # Étape 6 : Formattage et export
        topic_labels = self._extract_labels(snapshot)
        report = self._formatter.build_report(
            category=category,
            period=month,
            hybrid_results=hybrid_results,
            time_series_map=time_series_map,
            topic_labels=topic_labels,
        )

        if output_json_path:
            self._formatter.to_json(report, output_json_path)
            logger.info("Rapport JSON écrit : %s", output_json_path)

        if output_parquet_path:
            self._formatter.to_parquet(report, output_parquet_path)
            logger.info("Rapport Parquet écrit : %s", output_parquet_path)

        logger.info(
            "TrendPipeline terminé : %s/%s : %d topics, %d TRENDING, %d EMERGING, %d TRANSFORMING",
            category,
            month,
            report.n_topics_total,
            report.n_trending,
            report.n_emerging,
            report.n_transforming,
        )

        return report

    @staticmethod
    def _extract_labels(snapshot: SnapshotResult) -> dict[int, str]:
        """Extrait le mapping topic_id → label depuis le snapshot."""
        return dict(zip(snapshot.topic_id, snapshot.topic_label))
