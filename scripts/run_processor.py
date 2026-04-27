"""
scripts/run_processor.py

Point d'entrée CLI pour lancer le traitement arXiv RAW -> CLEAN.
Le backend de stockage (local ou MinIO) est lu depuis STORAGE_BACKEND dans .env.

Exemples :
    python scripts/run_processor.py --category stat_ML --year 2026 --month 2
    python scripts/run_processor.py --category cs_AI --year 2026
    python scripts/run_processor.py  # traite toutes les donnees RAW disponibles
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
from processor import ArxivProcessor, ProcessorConfig
from processor.storage import MinioProcessorStorage
from utils.logger import setup_logger
from utils.run_id import build_run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_processor",
        description="Pipeline Veille Scientifique — Processeur arXiv RAW -> CLEAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--category", "-c", default="",
        help="Categorie a traiter (ex: stat_ML). Vide = toutes.",
    )
    parser.add_argument(
        "--year", "-y", type=int, default=None,
        help="Annee a traiter (ex: 2026). Defaut : toutes.",
    )
    parser.add_argument(
        "--month", "-m", type=int, default=None,
        help="Mois a traiter 1-12 (ex: 2). Requiert --year.",
    )

    parser.add_argument(
        "--raw-path", default=ProcessorConfig.raw_data_path,
        help="Chemin zone RAW locale",
    )
    parser.add_argument(
        "--clean-path", default=ProcessorConfig.clean_data_path,
        help="Chemin zone CLEAN locale",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Ecraser les fichiers CLEAN existants",
    )
    parser.add_argument("--log-dir", default=ProcessorConfig.log_dir, help="Dossier de logs")
    parser.add_argument("--report-path", default=None, help="Chemin rapport JSON (optionnel)")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )

    return parser


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _build_storage(logger: logging.Logger) -> Optional[MinioProcessorStorage]:
    """Instancie le storage MinIO si STORAGE_BACKEND=minio, sinon None (local)."""
    if os.getenv("STORAGE_BACKEND", "local").lower() == "minio":
        storage = MinioProcessorStorage(
            endpoint=MINIO_CONFIG.endpoint,
            access_key=MINIO_CONFIG.access_key,
            secret_key=MINIO_CONFIG.secret_key,
            raw_bucket=MINIO_CONFIG.raw_bucket,
            clean_bucket=MINIO_CONFIG.clean_bucket,
            secure=MINIO_CONFIG.secure,
            logger=logger,
        )
        logger.info(
            "Storage MinIO : %s | RAW='%s' CLEAN='%s'",
            MINIO_CONFIG.endpoint, MINIO_CONFIG.raw_bucket, MINIO_CONFIG.clean_bucket,
        )
        return storage
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.month is not None and args.year is None:
        parser.error("--month requiert --year")

    cat_label = args.category or "all"
    run_id = build_run_id(f"process_{cat_label}")
    logger = setup_logger(
        name=f"arxiv.processor.{cat_label}",
        log_dir=args.log_dir,
        run_id=run_id,
        console_level=getattr(logging, args.log_level),
    )

    logger.info("Run ID : %s", run_id)
    logger.info("Storage : %s", os.getenv("STORAGE_BACKEND", "local"))

    backend = os.getenv("STORAGE_BACKEND", "local")

    try:
        config = ProcessorConfig(
            raw_data_path=args.raw_path,
            clean_data_path=args.clean_path,
            category=args.category,
            year=args.year,
            month=args.month,
            overwrite_existing=args.overwrite,
            log_dir=args.log_dir,
            storage_backend=backend,
        )
    except ValueError as exc:
        logger.error("Configuration invalide : %s", exc)
        return 1

    storage = _build_storage(logger)

    try:
        report = ArxivProcessor(config, logger=logger, storage=storage).run()
    except KeyboardInterrupt:
        logger.warning("Traitement interrompu manuellement.")
        return 0
    except Exception as exc:
        logger.exception("Erreur inattendue : %s", exc)
        return 1

    if args.report_path:
        report.save(Path(args.report_path))
        logger.info("Rapport sauvegarde : %s", args.report_path)

    logger.info("Traitement termine : status=%s", report.status)
    return 0 if report.status in ("success", "empty") else 1


if __name__ == "__main__":
    sys.exit(main())
