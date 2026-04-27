"""
dags/dag_bootstrap.py

DAG Bootstrap — Collecte initiale complète d'une catégorie arXiv.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param
from airflow.utils.trigger_rule import TriggerRule

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dags.pipeline_tasks import (
    task_collect,
    task_enrich_nlp,
    task_enrich_trends,
    task_process,
    task_sync_duckdb,
)

logger = logging.getLogger("arxiv.pipeline")


@dag(
    dag_id="arxiv_bootstrap",
    description=(
        "Collecte initiale complète d'une catégorie arXiv. "
        "Déclenché manuellement depuis le dashboard ou l'UI Airflow."
    ),
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["arxiv", "bootstrap"],
    params={
        "category_id": Param(
            default="",
            type="string",
            description="Identifiant interne de la catégorie (ex: stat_ML, cs_AI, math_AP).",
        ),
        "skip_collect": Param(
            default=False,
            type="boolean",
            description="Sauter l'étape de collecte (utile pour re-enrichir).",
        ),
        "skip_process": Param(
            default=False,
            type="boolean",
            description="Sauter l'étape RAW → CLEAN.",
        ),
        "skip_enrich": Param(
            default=False,
            type="boolean",
            description="Sauter les étapes NLP + détection de tendances.",
        ),
        "skip_dw": Param(
            default=False,
            type="boolean",
            description="Sauter la synchronisation dans dashboard.duckdb.",
        ),
    },
    doc_md=__doc__,
)
def bootstrap_dag():

    @task
    def load_config(params: dict = None) -> dict:
        from dashboard.core.config_store import ConfigStore

        category_id = (params or {}).get("category_id", "").strip()
        if not category_id:
            raise ValueError(
                "Le paramètre 'category_id' est obligatoire. "
                "Exemple : {'category_id': 'stat_ML'}"
            )

        store = ConfigStore()
        cfg = store.get(category_id)
        if cfg is None:
            raise ValueError(
                f"Catégorie '{category_id}' introuvable dans pipeline_configs.json. "
                "Configurez-la d'abord sur la Page 0 du dashboard."
            )

        logger.info(
            "[Bootstrap] Configuration chargée : %s (%s)", category_id, cfg.display_name
        )
        return {
            "category_id": category_id,
            "display_name": cfg.display_name,
            "lookback_days": cfg.lookback_days,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "max_results": cfg.max_results,
            "sort_by": cfg.sort_by,
            "sort_order": cfg.sort_order,
            "fetch_citations": cfg.fetch_citations,
            "nr_topics": cfg.nr_topics,
            "min_months_for_trend": cfg.min_months_for_trend,
            "z_score_window": cfg.z_score_window,
            "semantic_drift_enabled": cfg.semantic_drift_enabled,
            "skip_collect": (params or {}).get("skip_collect", False),
            "skip_process": (params or {}).get("skip_process", False),
            "skip_enrich": (params or {}).get("skip_enrich", False),
            "skip_dw": (params or {}).get("skip_dw", False),
        }

    @task
    def step_collect(config: dict) -> dict:
        if config.get("skip_collect"):
            logger.info("[Bootstrap] Collecte ignorée (skip_collect=True).")
            raise AirflowSkipException("skip_collect=True")
        logger.info("[Bootstrap] [1/5] Collecte : %s", config["category_id"])
        return task_collect(
            category_id=config["category_id"],
            lookback_days=config["lookback_days"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            max_results=config["max_results"],
            sort_by=config["sort_by"],
            sort_order=config["sort_order"],
            fetch_citations=config["fetch_citations"],
        )

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def step_process(config: dict) -> dict:
        if config.get("skip_process"):
            logger.info("[Bootstrap] Traitement ignoré (skip_process=True).")
            raise AirflowSkipException("skip_process=True")
        logger.info("[Bootstrap] [2/5] Traitement : %s", config["category_id"])
        return task_process(
            category_id=config["category_id"],
            year=None,
            month=None,
        )

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def step_enrich_nlp(config: dict) -> dict:
        if config.get("skip_enrich"):
            logger.info("[Bootstrap] Enrichissement ignoré (skip_enrich=True).")
            raise AirflowSkipException("skip_enrich=True")
        logger.info("[Bootstrap] [3/5] Enrichissement NLP : %s", config["category_id"])
        return task_enrich_nlp(
            category_id=config["category_id"],
            nr_topics=config["nr_topics"],
            year=None,
            month=None,
            skip_existing=False,
            force_refit=True,
        )

    @task
    def step_enrich_trends(config: dict) -> dict:
        logger.info("[Bootstrap] [4/5] Tendances : %s", config["category_id"])
        return task_enrich_trends(
            category_id=config["category_id"],
            semantic_drift_enabled=config["semantic_drift_enabled"],
            min_months_for_trend=config["min_months_for_trend"],
            z_score_window=config["z_score_window"],
            year=None,
            month=None,
        )

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def step_sync_duckdb(config: dict) -> dict:
        if config.get("skip_dw"):
            logger.info("[Bootstrap] Sync DW ignorée (skip_dw=True).")
            raise AirflowSkipException("skip_dw=True")
        logger.info("[Bootstrap] [5/5] Sync DuckDB : %s", config["category_id"])
        return task_sync_duckdb(
            category_id=config["category_id"],
            display_name=config["display_name"],
        )

    config = load_config()
    result_collect = step_collect(config)
    result_process = step_process(config)
    result_nlp = step_enrich_nlp(config)
    result_trends = step_enrich_trends(config)
    result_dw = step_sync_duckdb(config)

    result_collect >> result_process >> result_nlp >> result_trends >> result_dw


bootstrap_dag()
