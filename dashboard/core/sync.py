"""
dashboard/core/sync.py

Wrapper DuckDBSync pour le dashboard.

Encapsule l'appel à DuckDBSync.run() et retourne un résultat
structuré avec les compteurs post-sync et un message lisible.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Résultat d'une opération de synchronisation DuckDB."""

    category: str
    success: bool
    n_articles: int = 0
    n_topics: int = 0
    n_trend_reports: int = 0
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        """Retourne un message de résumé lisible pour le journal."""
        if not self.success:
            return f"Sync échouée : {self.error or 'erreur inconnue'}"
        parts = [f"{self.n_articles:,} articles".replace(",", " ")]
        if self.n_topics:
            parts.append(f"{self.n_topics} topics")
        if self.n_trend_reports:
            parts.append(f"{self.n_trend_reports} rapports de tendance")
        return "Sync OK · " + " · ".join(parts)


def run_sync(category: str, display_name: str) -> SyncResult:
    """
    Synchronise une catégorie vers dashboard.duckdb.

    Lit les variables d'environnement :
        DUCKDB_PATH   (défaut : data/dashboard.duckdb)
        GOLD_PATH     (défaut : data/GOLD)
        TRENDS_PATH   (défaut : data/TRENDS)

    Args:
        category     : Identifiant interne de la catégorie (ex: 'stat_ML').
        display_name : Nom lisible pour dim_categories.

    Returns:
        SyncResult avec les compteurs post-sync ou l'erreur.
    """
    duckdb_path = os.getenv("DUCKDB_PATH", "data/dashboard.duckdb")
    backend     = os.getenv("STORAGE_BACKEND", "local").lower()

    try:
        from duckdb_sync import DuckDBSync
        from duckdb_sync.reader import LocalReader, MinioReader

        if backend == "minio":
            reader = MinioReader.from_minio_config()
        else:
            reader = LocalReader(
                gold_path=os.getenv("GOLD_PATH", "data/GOLD"),
                trends_path=os.getenv("TRENDS_PATH", "data/TRENDS"),
            )

        sync = DuckDBSync(duckdb_path=duckdb_path, reader=reader)
        results = sync.run(
            categories=[category],
            display_names={category: display_name},
        )
        success = bool(results.get(category, False))

        if not success:
            return SyncResult(
                category=category,
                success=False,
                error="DuckDBSync a retourné False (voir logs serveur).",
            )

        # Récupérer les compteurs post-sync
        from dashboard.core.db import get_category_counts
        counts = get_category_counts(category)
        return SyncResult(
            category=category,
            success=True,
            n_articles=counts.n_articles,
            n_topics=counts.n_topics,
            n_trend_reports=counts.n_trend_reports,
        )

    except FileNotFoundError as exc:
        msg = str(exc)
        logger.warning("run_sync '%s' — fichiers absents : %s", category, msg)
        return SyncResult(category=category, success=False, error=msg)

    except ImportError as exc:
        msg = f"Module manquant : {exc}"
        logger.error("run_sync '%s' — import échoué : %s", category, msg)
        return SyncResult(category=category, success=False, error=msg)

    except Exception as exc:
        msg = str(exc)
        logger.error("run_sync '%s' échoué : %s", category, msg)
        return SyncResult(category=category, success=False, error=msg)
