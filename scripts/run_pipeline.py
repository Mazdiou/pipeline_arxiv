"""
scripts/run_pipeline.py

Lance le pipeline arXiv complet :
    1. Collecte      (arXiv API -> RAW)
    2. Traitement    (RAW -> CLEAN)
    3. Enrichissement (CLEAN -> GOLD -> TRENDS)
    4. Data Warehouse (GOLD + TRENDS -> dashboard.duckdb)

Le backend de stockage (local ou MinIO) est lu depuis STORAGE_BACKEND dans .env.

Exemples :
    python scripts/run_pipeline.py --category cs.AI --lookback-days 30
    python scripts/run_pipeline.py --category stat.ML --start-date 2026-01-01 --end-date 2026-03-31
    python scripts/run_pipeline.py --category cs.LG --skip-collect --year 2026 --month 3
    python scripts/run_pipeline.py --category cs.AI --skip-dw
    python scripts/run_pipeline.py --category cs.AI --skip-collect --skip-process --no-semantic
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from minio_config import MINIO_CONFIG  # déclenche load_dotenv()
from collector import ArxivCollector, CollectorConfig
from collector.storage import MinioStorage
from duckdb_sync import DuckDBSync
from enricher.config.settings import EnricherConfig
from enricher.config.trend_settings import TrendConfig
from enricher.pipeline.enricher import ArxivEnricher
from enricher.trend_detection import TrendPipeline, build_trend_storage
from enricher.trend_detection.output.formatter import TrendFormatter
from processor import ArxivProcessor, ProcessorConfig
from processor.storage import MinioProcessorStorage
from utils.logger import setup_logger
from utils.run_id import build_run_id

# Import des helpers GOLD partagés avec run_enricher
import io
import pandas as pd


def _watch_id(category: str) -> str:
    """Convertit une catégorie arXiv en watch_id (cs.AI -> cs_AI)."""
    return category.replace(".", "_")


def _minio_client():
    from minio import Minio
    return Minio(
        MINIO_CONFIG.endpoint,
        access_key=MINIO_CONFIG.access_key,
        secret_key=MINIO_CONFIG.secret_key,
        secure=MINIO_CONFIG.secure,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Pipeline complet arXiv : Collecte -> Traitement -> Enrichissement -> DW.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--category", "-c", required=True,
        help="Categorie arXiv (ex: cs.AI, stat.ML)",
    )

    collect_grp = parser.add_argument_group("Collecte arXiv")
    collect_grp.add_argument(
        "--lookback-days", type=int, default=CollectorConfig.lookback_days,
        help="Jours en arriere (ignore si --start-date fourni)",
    )
    collect_grp.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    collect_grp.add_argument("--end-date", default=None, help="YYYY-MM-DD (defaut: aujourd'hui)")
    collect_grp.add_argument(
        "--max-results", type=int, default=CollectorConfig.max_results,
        help="Nombre maximum d'articles a collecter",
    )

    filter_grp = parser.add_argument_group("Filtres de partitions (processor / enricher / DW)")
    filter_grp.add_argument(
        "--year", "-y", type=int, default=None,
        help="Annee a traiter (defaut: toutes)",
    )
    filter_grp.add_argument(
        "--month", "-m", type=int, default=None,
        help="Mois a traiter 1-12 (requiert --year)",
    )

    steps_grp = parser.add_argument_group("Controle des etapes")
    steps_grp.add_argument("--skip-collect", action="store_true", help="Sauter la collecte")
    steps_grp.add_argument("--skip-process", action="store_true", help="Sauter RAW -> CLEAN")
    steps_grp.add_argument(
        "--skip-enrich", action="store_true",
        help="Sauter l'enrichissement CLEAN -> GOLD -> TRENDS",
    )
    steps_grp.add_argument(
        "--skip-dw", action="store_true",
        help="Sauter la synchronisation dans le data warehouse DuckDB",
    )
    steps_grp.add_argument(
        "--no-semantic", action="store_true",
        help="Desactiver la derive semantique dans TrendPipeline (plus rapide)",
    )

    parser.add_argument("--log-dir", default="logs", help="Dossier de logs")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )

    return parser


# ---------------------------------------------------------------------------
# Étape 1 : Collecte
# ---------------------------------------------------------------------------


def _step_collect(args: argparse.Namespace, logger: logging.Logger) -> bool:
    log = logging.getLogger("arxiv.pipeline")
    log.info("=" * 60)
    log.info("ETAPE 1/4 — COLLECTE arXiv : %s", args.category)
    log.info("=" * 60)

    backend = os.getenv("STORAGE_BACKEND", "local").lower()
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
            category=args.category,
            lookback_days=args.lookback_days,
            start_date=args.start_date,
            end_date=args.end_date,
            max_results=args.max_results,
            log_dir=args.log_dir,
        )
    except ValueError as exc:
        log.error("Configuration collecte invalide : %s", exc)
        return False

    try:
        report = ArxivCollector(config, logger=logger, storage=storage).run()
    except Exception as exc:
        log.exception("Collecte echouee : %s", exc)
        return False

    ok = report.status in ("success", "empty", "partial")
    log.info(
        "Collecte terminee : %d articles | status=%s",
        report.total_collected, report.status,
    )
    return ok


# ---------------------------------------------------------------------------
# Étape 2 : Traitement
# ---------------------------------------------------------------------------


def _step_process(args: argparse.Namespace, logger: logging.Logger) -> bool:
    log = logging.getLogger("arxiv.pipeline")
    log.info("=" * 60)
    log.info("ETAPE 2/4 — TRAITEMENT RAW -> CLEAN : %s", args.category)
    log.info("=" * 60)

    backend = os.getenv("STORAGE_BACKEND", "local").lower()
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
            category=_watch_id(args.category),
            year=args.year,
            month=args.month,
            storage_backend=backend,
            log_dir=args.log_dir,
        )
    except ValueError as exc:
        log.error("Configuration processor invalide : %s", exc)
        return False

    try:
        report = ArxivProcessor(config, logger=logger, storage=storage).run()
    except Exception as exc:
        log.exception("Traitement echoue : %s", exc)
        return False

    ok = report.status in ("success", "empty")
    log.info("Traitement termine : status=%s", report.status)
    return ok


# ---------------------------------------------------------------------------
# Étape 3 : Enrichissement CLEAN → GOLD → TRENDS
# ---------------------------------------------------------------------------


def _step_enrich(args: argparse.Namespace, logger: logging.Logger) -> bool:
    log = logging.getLogger("arxiv.pipeline")
    log.info("=" * 60)
    log.info("ETAPE 3/4 — ENRICHISSEMENT CLEAN -> GOLD -> TRENDS : %s", args.category)
    log.info("=" * 60)

    category = _watch_id(args.category)
    backend = os.getenv("STORAGE_BACKEND", "local").lower()

    # 3a. CLEAN → GOLD
    log.info("  [3a] NLP CLEAN -> GOLD")
    try:
        enrich_config = EnricherConfig(
            categories=[category],
            years=[args.year] if args.year else None,
            months=[args.month] if args.month else None,
            storage_backend=backend,
        )
        enrich_report = ArxivEnricher(enrich_config, logger=logger).run()
    except Exception as exc:
        log.exception("Enrichissement GOLD echoue : %s", exc)
        return False

    if enrich_report.n_errors > 0:
        log.error("Enrichissement GOLD : %d erreur(s).", enrich_report.n_errors)
        return False

    log.info("  [3a] GOLD OK.")

    # 3b. GOLD → TRENDS
    log.info("  [3b] Detection de tendances GOLD -> TRENDS")

    years_filter = [args.year] if args.year else None
    months_filter = [args.month] if args.month else None

    if backend == "minio":
        partitions = _discover_gold_minio(category, years_filter, months_filter)
    else:
        partitions = _discover_gold_local("data/GOLD", category, years_filter, months_filter)

    if not partitions:
        log.warning("  Aucune partition GOLD pour trends — etape ignoree.")
        return True

    trend_config = TrendConfig(
        semantic_drift_enabled=not args.no_semantic,
        embeddings_cache_path="models/embeddings_cache",
        gold_data_path="data/GOLD",
    )
    trend_storage = build_trend_storage(storage_backend=backend, trends_path="data/TRENDS")
    pipeline = TrendPipeline(config=trend_config, storage=trend_storage)
    formatter = TrendFormatter()

    n_ok = n_err = 0
    for year, month in partitions:
        month_key = f"{year}-{month:02d}"
        try:
            if backend == "minio":
                gold_df = _read_gold_minio(category, year, month)
            else:
                gold_df = _read_gold_local("data/GOLD", category, year, month)
        except Exception as exc:
            log.error("  Lecture GOLD %s/%s : %s", category, month_key, exc)
            n_err += 1
            continue

        if gold_df.empty:
            log.warning("  GOLD vide pour %s/%s — ignore.", category, month_key)
            continue

        try:
            report = pipeline.run(gold_df=gold_df, category=category, month=month_key)
            if report is not None:
                report_df = formatter.to_dataframe(report)
                if not report_df.empty:
                    report_df = trend_storage.resolve_labels(report_df, category)
                    trend_storage.write_report(report_df, category, month_key)
                log.info(
                    "  OK %s : %d topics | %d TRENDING | %d EMERGING | %d TRANSFORMING",
                    month_key, report.n_topics_total,
                    report.n_trending, report.n_emerging, report.n_transforming,
                )
                n_ok += 1
        except Exception as exc:
            log.exception("  TrendPipeline %s/%s : %s", category, month_key, exc)
            n_err += 1

    log.info("  [3b] Trends : %d/%d partitions reussies.", n_ok, n_ok + n_err)
    return n_err == 0


# ---------------------------------------------------------------------------
# Étape 4 : Data Warehouse
# ---------------------------------------------------------------------------


def _step_dw(args: argparse.Namespace, logger: logging.Logger) -> bool:
    log = logging.getLogger("arxiv.pipeline")
    log.info("=" * 60)
    log.info("ETAPE 4/4 — SYNCHRONISATION DATA WAREHOUSE : %s", args.category)
    log.info("=" * 60)

    category = _watch_id(args.category)

    try:
        sync = DuckDBSync.from_env()
        results = sync.run(categories=[category])
    except Exception as exc:
        log.exception("Synchronisation DW echouee : %s", exc)
        return False

    ok = results.get(category, False)
    if ok:
        log.info("Synchronisation DW terminee avec succes.")
    else:
        log.error("Synchronisation DW echouee pour '%s'.", category)
    return ok


# ---------------------------------------------------------------------------
# Helpers GOLD
# ---------------------------------------------------------------------------


def _discover_gold_local(
    gold_path: str,
    category: str,
    years: list[int] | None,
    months: list[int] | None,
) -> list[tuple[int, int]]:
    root = Path(gold_path) / category
    if not root.exists():
        return []
    partitions: list[tuple[int, int]] = []
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            y = int(year_dir.name)
        except ValueError:
            continue
        if years and y not in years:
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            try:
                m = int(month_dir.name)
            except ValueError:
                continue
            if months and m not in months:
                continue
            if (month_dir / "articles.parquet").exists():
                partitions.append((y, m))
    return sorted(partitions)


def _discover_gold_minio(
    category: str,
    years: list[int] | None,
    months: list[int] | None,
) -> list[tuple[int, int]]:
    client = _minio_client()
    partitions: list[tuple[int, int]] = []
    try:
        objects = client.list_objects(
            MINIO_CONFIG.gold_bucket, prefix=f"{category}/", recursive=True
        )
        for obj in objects:
            key = obj.object_name
            if not key.endswith("articles.parquet"):
                continue
            parts = key.rstrip("/").split("/")
            if len(parts) < 4:
                continue
            try:
                m, y = int(parts[-2]), int(parts[-3])
            except ValueError:
                continue
            if years and y not in years:
                continue
            if months and m not in months:
                continue
            partitions.append((y, m))
    except Exception as exc:
        logging.getLogger("arxiv.pipeline").error(
            "Listing MinIO gold '%s' : %s", category, exc
        )
    return sorted(partitions)


def _read_gold_local(gold_path: str, category: str, year: int, month: int) -> pd.DataFrame:
    path = Path(gold_path) / category / str(year) / f"{month:02d}" / "articles.parquet"
    return pd.read_parquet(path)


def _read_gold_minio(category: str, year: int, month: int) -> pd.DataFrame:
    client = _minio_client()
    key = f"{category}/{year:04d}/{month:02d}/articles.parquet"
    response = None
    try:
        response = client.get_object(MINIO_CONFIG.gold_bucket, key)
        data = response.read()
    except Exception as exc:
        raise FileNotFoundError(f"MinIO gold : {key}") from exc
    finally:
        if response is not None:
            response.close()
            response.release_conn()
    return pd.read_parquet(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.month is not None and args.year is None:
        parser.error("--month requiert --year")

    run_id = build_run_id(f"pipeline_{_watch_id(args.category)}")
    console_level = getattr(logging, args.log_level)

    logger = setup_logger(
        name="arxiv.pipeline",
        log_dir=args.log_dir,
        run_id=run_id,
        console_level=console_level,
    )
    for pkg in ("collector", "processor", "enricher", "duckdb_sync"):
        setup_logger(name=pkg, log_dir=args.log_dir, run_id=run_id, console_level=console_level)

    log = logging.getLogger("arxiv.pipeline")
    log.info("Pipeline arXiv — Run ID : %s", run_id)
    log.info("Categorie : %s | Storage : %s", args.category, os.getenv("STORAGE_BACKEND", "local"))

    steps = [
        ("Collecte",        not args.skip_collect, lambda: _step_collect(args, logger)),
        ("Traitement",      not args.skip_process, lambda: _step_process(args, logger)),
        ("Enrichissement",  not args.skip_enrich,  lambda: _step_enrich(args, logger)),
        ("Data Warehouse",  not args.skip_dw,      lambda: _step_dw(args, logger)),
    ]

    try:
        for name, enabled, fn in steps:
            if not enabled:
                log.info("— %s : ignoree (--skip)", name)
                continue
            if not fn():
                log.error("Pipeline arrete a l'etape : %s", name)
                return 1
    except KeyboardInterrupt:
        log.warning("Pipeline interrompu manuellement.")
        return 130

    log.info("Pipeline termine avec succes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
