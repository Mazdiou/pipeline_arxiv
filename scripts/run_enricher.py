"""
scripts/run_enricher.py

Point d'entrée CLI pour lancer l'enrichissement complet :
    CLEAN -> GOLD (NLP : topics, novelty, entités, keywords)
    GOLD  -> TRENDS (détection de tendances et signaux faibles)

Le backend de stockage (local ou MinIO) est lu depuis STORAGE_BACKEND dans .env.
Les deux étapes utilisent automatiquement le même backend.

Exemples :
    python scripts/run_enricher.py --categories cs_AI
    python scripts/run_enricher.py --categories cs_AI stat_ML --years 2025 2026
    python scripts/run_enricher.py --categories math_AP --force --log-level DEBUG
    python scripts/run_enricher.py --categories cs_AI --no-semantic  # trend sans embeddings
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from minio_config import MINIO_CONFIG  # déclenche load_dotenv()
from enricher.config.settings import EnricherConfig
from enricher.config.trend_settings import TrendConfig
from enricher.pipeline.enricher import ArxivEnricher
from enricher.trend_detection import TrendPipeline, build_trend_storage
from enricher.trend_detection.output.formatter import TrendFormatter
from utils.logger import setup_logger
from utils.run_id import build_run_id

import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_enricher",
        description="Enrichissement NLP (CLEAN->GOLD) + Détection de tendances (GOLD->TRENDS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("Chemins (STORAGE_BACKEND=local uniquement)")
    paths.add_argument("--clean-path", default="data/CLEAN", help="Repertoire source CLEAN")
    paths.add_argument("--gold-path", default="data/GOLD", help="Repertoire destination GOLD")
    paths.add_argument("--trends-path", default="data/TRENDS", help="Repertoire destination TRENDS")
    paths.add_argument("--models-path", default="models", help="Dossier des modeles BERTopic")

    filters = parser.add_argument_group("Filtres de traitement")
    filters.add_argument(
        "--categories", nargs="*",
        help="Categories a traiter (ex: cs_AI stat_ML). Vide = toutes.",
    )
    filters.add_argument("--years", nargs="*", type=int, help="Annees a traiter (ex: 2025 2026)")
    filters.add_argument("--months", nargs="*", type=int, help="Mois a traiter 1-12")
    filters.add_argument(
        "--force", action="store_true",
        help="Reenrichir meme si la partition GOLD existe deja",
    )

    trend = parser.add_argument_group("Detection de tendances")
    trend.add_argument(
        "--no-semantic", action="store_true",
        help="Desactiver le calcul de derive semantique DST (plus rapide)",
    )
    trend.add_argument(
        "--skip-trends", action="store_true",
        help="Executer uniquement CLEAN->GOLD, sans l'etape GOLD->TRENDS",
    )

    log_grp = parser.add_argument_group("Logging")
    log_grp.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )
    log_grp.add_argument("--log-dir", default="logs")

    return parser


# ---------------------------------------------------------------------------
# Helpers GOLD (lecture partitions)
# ---------------------------------------------------------------------------


def _minio_client():
    """Retourne un client MinIO initialise depuis MINIO_CONFIG."""
    from minio import Minio
    return Minio(
        MINIO_CONFIG.endpoint,
        access_key=MINIO_CONFIG.access_key,
        secret_key=MINIO_CONFIG.secret_key,
        secure=MINIO_CONFIG.secure,
    )


def _discover_gold_partitions_local(
    gold_path: str,
    category: str,
    years: Optional[list[int]],
    months: Optional[list[int]],
) -> list[tuple[int, int]]:
    """Découvre les partitions GOLD disponibles sur le filesystem."""
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


def _discover_gold_partitions_minio(
    category: str,
    years: Optional[list[int]],
    months: Optional[list[int]],
) -> list[tuple[int, int]]:
    """Découvre les partitions GOLD disponibles dans le bucket MinIO gold."""
    client = _minio_client()
    prefix = f"{category}/"
    partitions: list[tuple[int, int]] = []

    try:
        objects = client.list_objects(MINIO_CONFIG.gold_bucket, prefix=prefix, recursive=True)
        for obj in objects:
            key: str = obj.object_name
            if not key.endswith("articles.parquet"):
                continue
            parts = key.rstrip("/").split("/")
            # Attendu : {category}/{YYYY}/{MM}/articles.parquet
            if len(parts) < 4:
                continue
            try:
                m = int(parts[-2])
                y = int(parts[-3])
            except ValueError:
                continue
            if years and y not in years:
                continue
            if months and m not in months:
                continue
            partitions.append((y, m))
    except Exception as exc:
        logging.getLogger("enricher.script").error(
            "Erreur listing MinIO '%s' (prefix '%s') : %s",
            MINIO_CONFIG.gold_bucket, prefix, exc,
        )

    return sorted(partitions)


def _read_gold_partition_local(gold_path: str, category: str, year: int, month: int) -> pd.DataFrame:
    path = Path(gold_path) / category / str(year) / f"{month:02d}" / "articles.parquet"
    return pd.read_parquet(path)


def _read_gold_partition_minio(category: str, year: int, month: int) -> pd.DataFrame:
    client = _minio_client()
    key = f"{category}/{year:04d}/{month:02d}/articles.parquet"
    response = None
    try:
        response = client.get_object(MINIO_CONFIG.gold_bucket, key)
        data = response.read()
    except Exception as exc:
        raise FileNotFoundError(
            f"Partition GOLD introuvable dans MinIO : {MINIO_CONFIG.gold_bucket}/{key}"
        ) from exc
    finally:
        if response is not None:
            response.close()
            response.release_conn()
    return pd.read_parquet(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Étape 1 : CLEAN → GOLD
# ---------------------------------------------------------------------------


def _step_enrich_gold(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Lance ArxivEnricher : CLEAN -> GOLD."""
    log = logging.getLogger("enricher.script")
    log.info("ETAPE 1/2 — Enrichissement NLP CLEAN -> GOLD")

    try:
        config = EnricherConfig(
            clean_path=args.clean_path,
            gold_path=args.gold_path,
            models_path=args.models_path,
            categories=args.categories,
            years=args.years,
            months=args.months,
            skip_existing=not args.force,
            storage_backend=os.getenv("STORAGE_BACKEND", "local"),
        )
        report = ArxivEnricher(config, logger=logger).run()
    except Exception as exc:
        log.exception("Enrichissement GOLD echoue : %s", exc)
        return False

    if report.n_errors > 0:
        log.error("Enrichissement termine avec %d erreur(s).", report.n_errors)
        return False

    log.info("Enrichissement GOLD termine avec succes.")
    return True


# ---------------------------------------------------------------------------
# Étape 2 : GOLD → TRENDS
# ---------------------------------------------------------------------------


def _step_detect_trends(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """
    Lance TrendPipeline partition par partition pour chaque catégorie.
    Gère automatiquement les backends local et MinIO.
    """
    log = logging.getLogger("enricher.script")
    log.info("ETAPE 2/2 — Detection de tendances GOLD -> TRENDS")

    backend = os.getenv("STORAGE_BACKEND", "local").lower()

    # Résoudre les catégories à traiter
    categories = args.categories or _discover_local_categories(args.gold_path, backend)
    if not categories:
        log.warning("Aucune categorie GOLD trouvee — trends ignorees.")
        return True

    config = TrendConfig(
        semantic_drift_enabled=not args.no_semantic,
        embeddings_cache_path=f"{args.models_path}/embeddings_cache",
        gold_data_path=args.gold_path,
    )
    storage = build_trend_storage(
        storage_backend=backend,
        trends_path=args.trends_path,
    )
    pipeline = TrendPipeline(config=config, storage=storage)
    formatter = TrendFormatter()

    n_ok = n_err = 0

    for category in categories:
        log.info("Trends : categorie '%s'", category)

        # Découverte des partitions GOLD
        if backend == "minio":
            partitions = _discover_gold_partitions_minio(category, args.years, args.months)
        else:
            partitions = _discover_gold_partitions_local(
                args.gold_path, category, args.years, args.months
            )

        if not partitions:
            log.warning("  Aucune partition GOLD pour '%s' — ignoree.", category)
            continue

        log.info("  %d partition(s) a analyser.", len(partitions))

        for year, month in partitions:
            month_key = f"{year}-{month:02d}"
            try:
                if backend == "minio":
                    gold_df = _read_gold_partition_minio(category, year, month)
                else:
                    gold_df = _read_gold_partition_local(args.gold_path, category, year, month)
            except Exception as exc:
                log.error("  Lecture GOLD echouee pour %s/%s : %s", category, month_key, exc)
                n_err += 1
                continue

            if gold_df.empty:
                log.warning("  DataFrame GOLD vide pour %s/%s — ignore.", category, month_key)
                continue

            try:
                report = pipeline.run(
                    gold_df=gold_df,
                    category=category,
                    month=month_key,
                )
                if report is not None:
                    report_df = formatter.to_dataframe(report)
                    if not report_df.empty:
                        report_df = storage.resolve_labels(report_df, category)
                        storage.write_report(report_df, category, month_key)
                    log.info(
                        "  OK %s : %d topics | %d TRENDING | %d EMERGING | %d TRANSFORMING",
                        month_key, report.n_topics_total,
                        report.n_trending, report.n_emerging, report.n_transforming,
                    )
                    n_ok += 1
                else:
                    log.warning("  Pipeline None pour %s/%s.", category, month_key)
            except Exception as exc:
                log.exception("  TrendPipeline echoue pour %s/%s : %s", category, month_key, exc)
                n_err += 1

    log.info(
        "Trends terminee : %d/%d partition(s) reussies.",
        n_ok, n_ok + n_err,
    )
    return n_err == 0


def _discover_local_categories(gold_path: str, backend: str) -> list[str]:
    """Retourne les catégories disponibles dans GOLD (local uniquement)."""
    if backend == "minio":
        return []
    root = Path(gold_path)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    cat_label = args.categories[0] if args.categories else "all"
    run_id = build_run_id(f"enrich_{cat_label}")

    logger = setup_logger(
        name="enricher",
        log_dir=args.log_dir,
        run_id=run_id,
        console_level=getattr(logging, args.log_level),
    )

    log = logging.getLogger("enricher.script")
    log.info("Run ID : %s", run_id)
    log.info("Categories : %s", ", ".join(args.categories) if args.categories else "toutes")
    log.info("Storage : %s", os.getenv("STORAGE_BACKEND", "local"))

    try:
        # Étape 1 : CLEAN → GOLD
        if not _step_enrich_gold(args, logger):
            log.error("Arret apres echec enrichissement GOLD.")
            return 1

        # Étape 2 : GOLD → TRENDS
        if not args.skip_trends:
            if not _step_detect_trends(args, logger):
                log.error("Arret apres echec detection de tendances.")
                return 1

        log.info("Run complet avec succes.")
        return 0

    except KeyboardInterrupt:
        log.warning("Interruption utilisateur.")
        return 130
    except Exception as exc:
        log.exception("Erreur fatale : %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
