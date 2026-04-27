"""
enricher/trend_detection/

Module de détection de tendances et de signaux faibles sur les topics arXiv.

Point d'entrée principal : TrendPipeline.run(gold_df, category, month)

Sous-modules :
    snapshot/      : Agrégation mensuelle des métriques par topic (TopicSnapshot)
    storage/       : Persistance des snapshots et de l'historique (BaseTrendStorage)
    core/          : Métriques statistiques et sémantiques (13 métriques)
    ingestion/     : Lecture de l'historique et agrégation des embeddings
    classification/: Classifieur 6 états + score composite
    comparison/    : 4 approches comparatives pour le mémoire (A/B/C/D)
    output/        : Formattage JSON/Parquet pour le dashboard
"""

from .pipeline import TrendPipeline
from .snapshot.partition_snapshot import TopicSnapshot
from .snapshot.snapshot_result import SnapshotResult
from .storage.trend_storage import (
    BaseTrendStorage,
    LocalTrendStorage,
    MinioTrendStorage,
    build_trend_storage,
)

__all__ = [
    "TrendPipeline",
    "TopicSnapshot",
    "SnapshotResult",
    "BaseTrendStorage",
    "LocalTrendStorage",
    "MinioTrendStorage",
    "build_trend_storage",
]
