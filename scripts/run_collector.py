"""
scripts/run_collector.py

Point d'entrée CLI pour lancer la collecte arXiv.
Le backend de stockage (local ou MinIO) est lu depuis STORAGE_BACKEND dans .env.

Exemples :
    python scripts/run_collector.py --category cs.AI --lookback-days 30
    python scripts/run_collector.py --category stat.ML --query "transformer" --max-results 1000
    python scripts/run_collector.py --category q-bio.NC --start-date 2024-01-01 --end-date 2024-06-30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from minio_config import MINIO_CONFIG  # déclenche load_dotenv()
from collector import ArxivCollector, CollectorConfig
from collector.storage import MinioStorage
from utils.logger import setup_logger
from utils.run_id import build_run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_collector",
        description="Pipeline Veille Scientifique — Collecteur arXiv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--category", "-c", default="", help="Categorie arXiv (ex: cs.AI)")
    parser.add_argument("--query", "-q", default="", help="Mots-cles complementaires")

    parser.add_argument(
        "--lookback-days", "-d", type=int, default=CollectorConfig.lookback_days,
        help="Jours en arriere (ignore si --start-date fourni)",
    )
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD (prioritaire sur lookback)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (defaut: aujourd'hui)")

    parser.add_argument(
        "--max-results", "-n", type=int, default=CollectorConfig.max_results,
        help="Nombre maximum d'articles",
    )
    parser.add_argument(
        "--batch-size", type=int, default=CollectorConfig.batch_size,
        help="Taille des lots pour pagination",
    )
    parser.add_argument(
        "--delay", type=float, default=CollectorConfig.delay_between_batches,
        help="Delai entre appels API (sec, min recommande 4.0)",
    )
    parser.add_argument(
        "--sort-by", default=CollectorConfig.sort_by,
        choices=["submittedDate", "relevance", "lastUpdatedDate"],
    )
    parser.add_argument(
        "--sort-order", default=CollectorConfig.sort_order,
        choices=["descending", "ascending"],
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Ecraser fichiers existants (sinon fusion/deduplication)",
    )

    parser.add_argument("--raw-path", default=CollectorConfig.raw_data_path, help="Chemin zone RAW")
    parser.add_argument("--log-dir", default=CollectorConfig.log_dir, help="Dossier de logs")
    parser.add_argument("--report-path", default=None, help="Chemin pour sauvegarder le rapport JSON")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )

    return parser


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _build_storage(logger: logging.Logger) -> Optional[MinioStorage]:
    """Instancie le storage MinIO si STORAGE_BACKEND=minio, sinon None (local)."""
    if os.getenv("STORAGE_BACKEND", "local").lower() == "minio":
        storage = MinioStorage(
            endpoint=MINIO_CONFIG.endpoint,
            access_key=MINIO_CONFIG.access_key,
            secret_key=MINIO_CONFIG.secret_key,
            bucket=MINIO_CONFIG.raw_bucket,
            secure=MINIO_CONFIG.secure,
            overwrite=MINIO_CONFIG.overwrite,
            logger=logger,
        )
        logger.info("Storage MinIO : %s / bucket '%s'", MINIO_CONFIG.endpoint, MINIO_CONFIG.raw_bucket)
        return storage
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    run_id = build_run_id(f"collect_{args.category or args.query or 'all'}")
    logger = setup_logger(
        name=f"arxiv.collector.{args.category or 'all'}",
        log_dir=args.log_dir,
        run_id=run_id,
        console_level=getattr(logging, args.log_level),
    )

    logger.info("Run ID : %s", run_id)
    logger.info("Storage : %s", os.getenv("STORAGE_BACKEND", "local"))

    try:
        config = CollectorConfig(
            category=args.category,
            query=args.query,
            lookback_days=args.lookback_days,
            start_date=args.start_date,
            end_date=args.end_date,
            max_results=args.max_results,
            batch_size=args.batch_size,
            delay_between_batches=args.delay,
            sort_by=args.sort_by,
            sort_order=args.sort_order,
            overwrite_existing=args.overwrite,
            raw_data_path=args.raw_path,
            log_dir=args.log_dir,
        )
    except ValueError as exc:
        logger.error("Configuration invalide : %s", exc)
        return 1

    storage = _build_storage(logger)

    try:
        report = ArxivCollector(config, logger=logger, storage=storage, run_id=run_id).run()
    except KeyboardInterrupt:
        logger.warning("Collecte interrompue manuellement.")
        return 0
    except Exception as exc:
        logger.exception("Erreur inattendue : %s", exc)
        return 1

    if args.report_path:
        report.save(Path(args.report_path))
        logger.info("Rapport sauvegarde : %s", args.report_path)

    logger.info(
        "Collecte terminee : %d articles | status=%s", report.total_collected, report.status
    )
    return 0 if report.status in ("success", "empty", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
