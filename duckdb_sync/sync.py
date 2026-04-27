"""
DuckDBSync : orchestrateur principal du data mart dashboard.duckdb.

Séquence pour chaque catégorie (dans une transaction unique) :
    1. delete_category()       : purge les anciennes données
    2. load_trends()           : fact_trend_reports + labels descriptifs
    3. load_topics()           : dim_topics (avec labels) + fact_topic_monthly
    4. load_articles()         : fact_articles
    5. upsert_dim_categories() : métadonnées de la catégorie

"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

from .loaders.articles import load_articles
from .loaders.topics import load_topics
from .loaders.trends import load_trends
from .reader import BaseDataReader, LocalReader, MinioReader
from .schema import delete_category, init_schema

logger = logging.getLogger(__name__)


class DuckDBSync:
    """
    Synchronise GOLD + TRENDS vers dashboard.duckdb.

    Args:
        duckdb_path : Chemin vers le fichier dashboard.duckdb.
        reader      : Reader abstrait (LocalReader ou MinioReader).
        gold_path   : Chemin GOLD local (utilisé si reader est None).
        trends_path : Chemin TRENDS local (utilisé si reader est None).
    """

    def __init__(
        self,
        duckdb_path: str = "data/dashboard.duckdb",
        reader: Optional[BaseDataReader] = None,
        gold_path: str = "data/GOLD",
        trends_path: str = "data/TRENDS",
    ) -> None:
        self._duckdb_path = duckdb_path
        self._reader = reader or LocalReader(gold_path, trends_path)

    @classmethod
    def from_env(cls) -> "DuckDBSync":
        """
        Construit DuckDBSync depuis les variables d'environnement.

        Variables :
            DUCKDB_PATH     : chemin dashboard.duckdb  (défaut: data/dashboard.duckdb)
            STORAGE_BACKEND : 'local' ou 'minio'       (défaut: local)
            GOLD_PATH       : chemin GOLD local        (défaut: data/GOLD)
            TRENDS_PATH     : chemin TRENDS local      (défaut: data/TRENDS)
        """
        duckdb_path = os.getenv("DUCKDB_PATH", "data/dashboard.duckdb")
        backend = os.getenv("STORAGE_BACKEND", "local").lower()

        if backend == "minio":
            reader: BaseDataReader = MinioReader.from_minio_config()
            logger.info("Backend MinIO activé.")
        else:
            reader = LocalReader(
                gold_path=os.getenv("GOLD_PATH", "data/GOLD"),
                trends_path=os.getenv("TRENDS_PATH", "data/TRENDS"),
            )
            logger.info("Backend local activé.")

        return cls(duckdb_path=duckdb_path, reader=reader)

    def run(
        self,
        categories: Optional[list[str]] = None,
        display_names: Optional[dict[str, str]] = None,
    ) -> dict[str, bool]:
        """
        Synchronise une ou plusieurs catégories vers dashboard.duckdb.

        Args:
            categories    : Catégories à synchroniser. None = toutes celles disponibles dans GOLD.
            display_names : dict[category : display_name] pour dim_categories.

        Returns:
            dict[category : success].
        """
        Path(self._duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(self._duckdb_path)
        try:
            init_schema(conn)
            categories_to_sync = categories or self._reader.discover_categories()

            if not categories_to_sync:
                logger.warning("Aucune catégorie trouvée dans GOLD.")
                return {}

            results: dict[str, bool] = {}
            for category in categories_to_sync:
                display_name = (display_names or {}).get(category, category)
                results[category] = self._sync_category(conn, category, display_name)

            logger.info(
                "Sync terminé : %d/%d catégories réussies.",
                sum(results.values()),
                len(results),
            )
            return results
        finally:
            conn.close()

    def _sync_category(
        self,
        conn: duckdb.DuckDBPyConnection,
        category: str,
        display_name: str,
    ) -> bool:
        """
        Synchronise une catégorie complète dans une transaction atomique.

        Ordre critique :
            load_trends() : fournit topic_labels
            load_topics() : consomme topic_labels pour dim_topics
            load_articles() : indépendant des deux précédents

        Returns:
            True si sync réussie, False si erreur (rollback effectué).
        """
        logger.info("Sync catégorie : '%s'", category)
        try:
            conn.execute("BEGIN")
            delete_category(conn, category)

            # Rapports : labels descriptifs extraits pour dim_topics
            topic_labels = load_trends(conn, category, self._reader)

            # Historique : dim_topics enrichi avec labels + fact_topic_monthly
            load_topics(conn, category, self._reader, topic_labels)

            # Articles GOLD : fact_articles
            load_articles(conn, category, self._reader)

            self._upsert_dim_categories(conn, category, display_name)
            conn.execute("COMMIT")
            logger.info("Sync '%s' terminée.", category)
            return True

        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error("Sync '%s' échouée (rollback) : %s", category, exc)
            return False

    def _upsert_dim_categories(
        self,
        conn: duckdb.DuckDBPyConnection,
        category: str,
        display_name: str,
    ) -> None:
        """
        Insère ou met à jour la ligne dans dim_categories.

        Conserve bootstrapped_at si la catégorie existe déjà.
        Met toujours à jour last_synced_at avec l'heure courante UTC.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        existing = conn.execute(
            "SELECT bootstrapped_at FROM dim_categories WHERE category = ?",
            [category],
        ).fetchone()
        bootstrapped_at = existing[0] if existing else now

        conn.execute(
            """
            INSERT INTO dim_categories (category, display_name, bootstrapped_at, last_synced_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (category) DO UPDATE SET
                display_name   = excluded.display_name,
                last_synced_at = excluded.last_synced_at
            """,
            [category, display_name, bootstrapped_at, now],
        )
