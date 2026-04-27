"""
dags/pipeline_tasks.py

Fonctions Python exécutées par les tâches Airflow.

Corrections appliquées :
    - task_enrich_nlp   : paramètre force_refit restauré (était silencieusement supprimé)
    - task_enrich_nlp   : TopicConfig instanciée de façon unifiée
    - task_enrich_trends : partition vide comptée comme n_ok (plus de faux ratio)
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger("arxiv.pipeline")


# =========================================================================== #
# Utilitaires internes                                                         #
# =========================================================================== #


def _to_arxiv_id(category_id: str) -> str:
    return category_id.replace("_", ".", 1)


def _minio_client():
    from minio import Minio
    from minio_config import MINIO_CONFIG

    return Minio(
        MINIO_CONFIG.endpoint,
        access_key=MINIO_CONFIG.access_key,
        secret_key=MINIO_CONFIG.secret_key,
        secure=MINIO_CONFIG.secure,
    )


def _discover_gold_partitions(
    category_id: str,
    year_filter: Optional[int] = None,
    month_filter: Optional[int] = None,
) -> list[tuple[int, int]]:
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "minio":
        return _discover_gold_minio(category_id, year_filter, month_filter)
    return _discover_gold_local(category_id, year_filter, month_filter)


def _discover_gold_local(
    category_id: str,
    year_filter: Optional[int] = None,
    month_filter: Optional[int] = None,
) -> list[tuple[int, int]]:
    gold_root = Path(os.getenv("GOLD_PATH", "data/GOLD")) / category_id
    if not gold_root.exists():
        return []
    partitions: list[tuple[int, int]] = []
    for year_dir in sorted(gold_root.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        if year_filter is not None and year != year_filter:
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            try:
                month = int(month_dir.name)
            except ValueError:
                continue
            if month_filter is not None and month != month_filter:
                continue
            if (month_dir / "articles.parquet").exists():
                partitions.append((year, month))
    return sorted(partitions)


def _discover_gold_minio(
    category_id: str,
    year_filter: Optional[int] = None,
    month_filter: Optional[int] = None,
) -> list[tuple[int, int]]:
    from minio_config import MINIO_CONFIG

    client = _minio_client()
    partitions: list[tuple[int, int]] = []
    try:
        objects = client.list_objects(
            MINIO_CONFIG.gold_bucket,
            prefix=f"{category_id}/",
            recursive=True,
        )
        for obj in objects:
            key = obj.object_name
            if not key.endswith("articles.parquet"):
                continue
            parts = key.rstrip("/").split("/")
            if len(parts) < 4:
                continue
            try:
                year = int(parts[-3])
                month = int(parts[-2])
            except ValueError:
                continue
            if year_filter is not None and year != year_filter:
                continue
            if month_filter is not None and month != month_filter:
                continue
            partitions.append((year, month))
    except Exception as exc:
        logger.error("Listing GOLD MinIO '%s' : %s", category_id, exc)
    return sorted(partitions)


def _read_gold_partition(category_id: str, year: int, month: int):
    import pandas as pd

    backend = os.getenv("STORAGE_BACKEND", "local").lower()

    if backend == "minio":
        from minio_config import MINIO_CONFIG

        client = _minio_client()
        key = f"{category_id}/{year:04d}/{month:02d}/articles.parquet"
        response = None
        try:
            response = client.get_object(MINIO_CONFIG.gold_bucket, key)
            data = response.read()
        except Exception as exc:
            raise FileNotFoundError(
                f"Partition GOLD introuvable dans MinIO : {key}"
            ) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()
        return pd.read_parquet(io.BytesIO(data))
    else:
        path = (
            Path(os.getenv("GOLD_PATH", "data/GOLD"))
            / category_id
            / str(year)
            / f"{month:02d}"
            / "articles.parquet"
        )
        return pd.read_parquet(path)


# =========================================================================== #
# Tâche 1 — Collecte arXiv API → RAW                                          #
# =========================================================================== #


def task_collect(
    category_id: str,
    lookback_days: int = 180,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 6000,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    fetch_citations: bool = True,
) -> dict:
    arxiv_id = _to_arxiv_id(category_id)
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    logger.info(
        "[Collecte] %s | backend=%s | lookback=%sd", arxiv_id, backend, lookback_days
    )

    from collector import ArxivCollector, CollectorConfig
    from collector.storage import MinioStorage
    from minio_config import MINIO_CONFIG

    storage = None
    if backend == "minio":
        storage = MinioStorage(
            endpoint=MINIO_CONFIG.endpoint,
            access_key=MINIO_CONFIG.access_key,
            secret_key=MINIO_CONFIG.secret_key,
            bucket=MINIO_CONFIG.raw_bucket,
            secure=MINIO_CONFIG.secure,
            logger=logger,
        )

    try:
        config = CollectorConfig(
            category=arxiv_id,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            fetch_citations=fetch_citations,
        )
    except ValueError as exc:
        raise RuntimeError(f"[Collecte] Configuration invalide : {exc}") from exc

    try:
        report = ArxivCollector(config, logger=logger, storage=storage).run()
    except Exception as exc:
        raise RuntimeError(f"[Collecte] Échoué : {exc}") from exc

    if report.status not in ("success", "empty", "partial"):
        raise RuntimeError(f"[Collecte] Status inattendu : {report.status}")

    result = {"n_collected": report.total_collected, "status": report.status}
    logger.info("[Collecte] Terminée : %s", result)
    return result


# =========================================================================== #
# Tâche 2 — Traitement RAW → CLEAN                                            #
# =========================================================================== #


def task_process(
    category_id: str,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    logger.info("[Traitement] %s | year=%s month=%s", category_id, year, month)

    from processor import ArxivProcessor, ProcessorConfig
    from processor.storage import MinioProcessorStorage
    from minio_config import MINIO_CONFIG

    storage = None
    if backend == "minio":
        storage = MinioProcessorStorage(
            endpoint=MINIO_CONFIG.endpoint,
            access_key=MINIO_CONFIG.access_key,
            secret_key=MINIO_CONFIG.secret_key,
            raw_bucket=MINIO_CONFIG.raw_bucket,
            clean_bucket=MINIO_CONFIG.clean_bucket,
            secure=MINIO_CONFIG.secure,
            logger=logger,
        )

    try:
        config = ProcessorConfig(
            category=category_id,
            year=year,
            month=month,
            storage_backend=backend,
        )
    except ValueError as exc:
        raise RuntimeError(f"[Traitement] Configuration invalide : {exc}") from exc

    try:
        report = ArxivProcessor(config, logger=logger, storage=storage).run()
    except Exception as exc:
        raise RuntimeError(f"[Traitement] Échoué : {exc}") from exc

    if report.status not in ("success", "empty"):
        raise RuntimeError(f"[Traitement] Status inattendu : {report.status}")

    result = {"status": report.status}
    logger.info("[Traitement] Terminé : %s", result)
    return result


# =========================================================================== #
# Tâche 3a — Enrichissement NLP CLEAN → GOLD                                  #
# =========================================================================== #


def task_enrich_nlp(
    category_id: str,
    nr_topics: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    skip_existing: bool = True,
    force_refit: bool = False,  # CORRECTION : paramètre restauré
) -> dict:
    """
    Enrichit les articles CLEAN avec NLP et écrit dans GOLD.

    Args:
        force_refit : True → reconstruire le modèle BERTopic depuis zéro.
                      Utilisé par le bootstrap initial (skip_existing=False).
    """
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    logger.info(
        "[Enrichissement NLP] %s | year=%s month=%s nr_topics=%s skip=%s force_refit=%s",
        category_id,
        year,
        month,
        nr_topics,
        skip_existing,
        force_refit,
    )

    from enricher.config.settings import EnricherConfig
    from enricher.config.topic_settings import TopicConfig
    from enricher.pipeline.enricher import ArxivEnricher

    # CORRECTION : instanciation unifiée de TopicConfig
    topic_cfg = TopicConfig(
        nr_topics=nr_topics if nr_topics is not None else TopicConfig().nr_topics,
        force_refit=force_refit,
    )

    try:
        config = EnricherConfig(
            categories=[category_id],
            years=[year] if year is not None else None,
            months=[month] if month is not None else None,
            storage_backend=backend,
            skip_existing=skip_existing,
            topic=topic_cfg,
        )
        report = ArxivEnricher(config, logger=logger).run()
    except Exception as exc:
        raise RuntimeError(f"[Enrichissement NLP] Échoué : {exc}") from exc

    if report.n_errors > 0:
        raise RuntimeError(
            f"[Enrichissement NLP] {report.n_errors} erreur(s) rencontrée(s)."
        )

    result = {"n_errors": 0}
    logger.info("[Enrichissement NLP] Terminé : %s", result)
    return result


# =========================================================================== #
# Tâche 3b — Détection de tendances GOLD → TRENDS                             #
# =========================================================================== #


def task_enrich_trends(
    category_id: str,
    semantic_drift_enabled: bool = True,
    min_months_for_trend: int = 4,
    z_score_window: int = 6,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    logger.info(
        "[Tendances] %s | year=%s month=%s semantic=%s",
        category_id,
        year,
        month,
        semantic_drift_enabled,
    )

    partitions = _discover_gold_partitions(category_id, year, month)
    if not partitions:
        logger.warning(
            "[Tendances] Aucune partition GOLD pour '%s' — étape ignorée.", category_id
        )
        return {"n_processed": 0, "n_errors": 0}

    from enricher.config.trend_settings import TrendConfig
    from enricher.trend_detection import TrendPipeline, build_trend_storage
    from enricher.trend_detection.output.formatter import TrendFormatter

    trend_config = TrendConfig(
        semantic_drift_enabled=semantic_drift_enabled,
        min_months_for_trend=min_months_for_trend,
        z_score_window=z_score_window,
    )
    trend_storage = build_trend_storage(
        storage_backend=backend,
        trends_path=os.getenv("TRENDS_PATH", "data/TRENDS"),
    )
    pipeline = TrendPipeline(config=trend_config, storage=trend_storage)
    formatter = TrendFormatter()

    n_ok = n_err = 0
    for part_year, part_month in partitions:
        month_key = f"{part_year}-{part_month:02d}"
        try:
            gold_df = _read_gold_partition(category_id, part_year, part_month)
        except Exception as exc:
            logger.error("[Tendances] Lecture GOLD %s : %s", month_key, exc)
            n_err += 1
            continue

        # CORRECTION : partition vide = ignorée mais comptée OK (ne fausse pas le ratio)
        if gold_df.empty:
            logger.warning("[Tendances] GOLD vide pour %s — ignoré.", month_key)
            n_ok += 1
            continue

        try:
            report = pipeline.run(
                gold_df=gold_df, category=category_id, month=month_key
            )
            if report is not None:
                report_df = formatter.to_dataframe(report)
                if not report_df.empty:
                    report_df = trend_storage.resolve_labels(report_df, category_id)
                    trend_storage.write_report(report_df, category_id, month_key)
            n_ok += 1
            logger.info(
                "[Tendances] %s OK : %d topics | %d trending | %d emerging",
                month_key,
                getattr(report, "n_topics_total", 0),
                getattr(report, "n_trending", 0),
                getattr(report, "n_emerging", 0),
            )
        except Exception as exc:
            logger.error("[Tendances] Partition %s : %s", month_key, exc)
            n_err += 1

    result = {"n_processed": n_ok, "n_errors": n_err}
    logger.info("[Tendances] Terminé : %s", result)

    if n_err > 0:
        raise RuntimeError(
            f"[Tendances] {n_err} partition(s) en erreur sur {n_ok + n_err}."
        )
    return result


# =========================================================================== #
# Tâche 4 — Synchronisation Data Warehouse                                    #
# =========================================================================== #


def task_sync_duckdb(category_id: str, display_name: str) -> dict:
    logger.info("[Sync DuckDB] %s (%s)", category_id, display_name)

    try:
        from duckdb_sync import DuckDBSync

        sync = DuckDBSync.from_env()
        results = sync.run(
            categories=[category_id],
            display_names={category_id: display_name},
        )
    except Exception as exc:
        raise RuntimeError(f"[Sync DuckDB] Échoué : {exc}") from exc

    success = bool(results.get(category_id, False))
    if not success:
        raise RuntimeError(
            f"[Sync DuckDB] DuckDBSync a retourné False pour '{category_id}'."
        )

    result = {"success": True}
    logger.info("[Sync DuckDB] Terminé : %s", result)
    return result
