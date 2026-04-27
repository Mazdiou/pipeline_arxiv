"""
scripts/run_dw.py

Point d'entrée CLI pour synchroniser une catégorie dans dashboard.duckdb.

Lit GOLD + TRENDS (local ou MinIO selon STORAGE_BACKEND dans .env) et charge
les données dans le data mart DuckDB via le module duckdb_sync.

Exemples :
    python scripts/run_dw.py --category math_AP
    python scripts/run_dw.py --category cs_AI --display-name "Computer Science - AI"
    python scripts/run_dw.py --category stat_ML cs_AI math_AP  # plusieurs catégories
    python scripts/run_dw.py  # toutes les catégories disponibles dans GOLD
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

from minio_config import MINIO_CONFIG  # noqa: F401 — déclenche load_dotenv()
from duckdb_sync import DuckDBSync
from utils.logger import setup_logger
from utils.run_id import build_run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_dw",
        description="Synchronisation GOLD + TRENDS vers dashboard.duckdb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--category", "-c", nargs="*", dest="categories",
        help="Categorie(s) a synchroniser (ex: math_AP cs_AI). Vide = toutes.",
    )
    parser.add_argument(
        "--display-name", "-n", nargs="*", dest="display_names",
        metavar="CATEGORY=NAME",
        help=(
            "Noms d'affichage pour dim_categories, format 'cat=Nom' "
            "(ex: math_AP='Mathematics - AP'). Optionnel."
        ),
    )
    parser.add_argument(
        "--duckdb-path",
        default=os.getenv("DUCKDB_PATH", "data/dashboard.duckdb"),
        help="Chemin vers le fichier dashboard.duckdb",
    )

    parser.add_argument("--log-dir", default="logs", help="Dossier de logs")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )

    return parser


# ---------------------------------------------------------------------------
# Parsing des display names
# ---------------------------------------------------------------------------


def _parse_display_names(raw: list[str] | None) -> dict[str, str]:
    """
    Convertit ['math_AP=Mathematics - AP', 'cs_AI=CS - AI'] en dict.
    Les entrées sans '=' sont ignorées.
    """
    if not raw:
        return {}
    result: dict[str, str] = {}
    for item in raw:
        if "=" in item:
            cat, _, name = item.partition("=")
            result[cat.strip()] = name.strip()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    categories = args.categories or None
    cat_label = categories[0] if categories and len(categories) == 1 else "multi"
    run_id = build_run_id(f"dw_{cat_label}")

    logger = setup_logger(
        name="duckdb_sync",
        log_dir=args.log_dir,
        run_id=run_id,
        console_level=getattr(logging, args.log_level),
    )

    log = logging.getLogger("duckdb_sync.script")
    log.info("Run ID : %s", run_id)
    log.info("Storage : %s", os.getenv("STORAGE_BACKEND", "local"))
    log.info(
        "Categories : %s",
        ", ".join(categories) if categories else "toutes (depuis GOLD)",
    )
    log.info("DuckDB : %s", args.duckdb_path)

    display_names = _parse_display_names(args.display_names)

    # Surcharge le chemin DuckDB si passé en argument
    os.environ["DUCKDB_PATH"] = args.duckdb_path

    try:
        sync = DuckDBSync.from_env()
        results = sync.run(
            categories=categories,
            display_names=display_names if display_names else None,
        )
    except KeyboardInterrupt:
        log.warning("Synchronisation interrompue manuellement.")
        return 130
    except Exception as exc:
        log.exception("Erreur fatale lors de la synchronisation : %s", exc)
        return 1

    if not results:
        log.warning("Aucune categorie synchronisee.")
        return 1

    n_ok = sum(results.values())
    n_total = len(results)

    for category, success in results.items():
        status = "OK" if success else "FAIL"
        log.info("  [%s] %s", status, category)

    log.info("Sync terminee : %d/%d categories reussies.", n_ok, n_total)
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
