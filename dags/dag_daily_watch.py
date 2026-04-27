"""
DAG Daily Watch : Collecte et traitement quotidiens de toutes les catégories.

Planification :
    Chaque jour à 2h00 UTC : schedule="0 2 * * *"

Comportement :
    1. Lit la liste de toutes les catégories dans pipeline_configs.json.
    2. Pour chaque catégorie, exécute en parallèle :
        a. Collecte  : derniers N jours (ARXIV_DAILY_LOOKBACK_DAYS, défaut : 7)
        b. Traitement : partition du mois courant uniquement (RAW → CLEAN)

    L'enrichissement NLP, la détection de tendances et la synchronisation DuckDB
    sont intentionnellement exclus : ces étapes sont déléguées à
    dag_monthly_enrich qui s'exécute automatiquement le 1er de chaque mois.

Airflow Variable optionnelle :
    ARXIV_DAILY_LOOKBACK_DAYS (int) : jours de collecte incrémentale (défaut : 7)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from airflow.decorators import dag, task

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dags.pipeline_tasks import (
    task_collect,
    task_process,
)

# Fenêtre glissante de collecte. 7 jours absorbent les délais de publication
# arXiv (certains articles apparaissent 2-3 jours après soumission) et couvrent
# les week-ends où la collecte ne change pas.
_DEFAULT_DAILY_LOOKBACK_DAYS = 7


# DAG principal : collecte + traitement quotidien pour toutes les catégories
@dag(
    dag_id="arxiv_daily_watch",
    description=(
        "Mise à jour quotidienne incrémentale de toutes les catégories arXiv configurées."
    ),
    schedule="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # Un seul run actif à la fois
    max_active_tasks=2,  # Max 2 catégories traitées en parallèle
    tags=["arxiv", "watch", "daily"],
    doc_md=__doc__,
)
def daily_watch_dag():

    # Chargement des catégories
    @task
    def get_active_categories() -> list[dict]:
        """
        Lit toutes les catégories configurées depuis pipeline_configs.json.

        Retourne une liste de dicts : un dict par catégorie : contenant
        tous les paramètres nécessaires au pipeline incrémental.
        Si aucune catégorie n'est configurée, retourne une liste vide
        (les tâches en aval seront sautées automatiquement via expand).
        """
        import logging
        from airflow.models import Variable
        from dashboard.core.config_store import ConfigStore

        log = logging.getLogger("arxiv.pipeline")

        daily_lookback = int(
            Variable.get(
                "ARXIV_DAILY_LOOKBACK_DAYS",
                default_var=str(_DEFAULT_DAILY_LOOKBACK_DAYS),
            )
        )

        store = ConfigStore()
        configs = store.load_all()

        if not configs:
            log.warning(
                "[Daily Watch] Aucune catégorie dans pipeline_configs.json : "
                "rien à faire."
            )
            return []

        now = datetime.now(timezone.utc)
        current_year = now.year
        current_month = now.month

        category_params = []
        for cat_id, cfg in configs.items():
            category_params.append(
                {
                    "category_id": cat_id,
                    "lookback_days": daily_lookback,
                    "max_results": cfg.max_results,
                    "sort_by": cfg.sort_by,
                    "sort_order": cfg.sort_order,
                    "fetch_citations": cfg.fetch_citations,
                    # Filtre temporel : mois courant uniquement
                    "current_year": current_year,
                    "current_month": current_month,
                }
            )

        log.info(
            "[Daily Watch] %d catégorie(s) à mettre à jour : %s",
            len(category_params),
            [p["category_id"] for p in category_params],
        )
        return category_params

    # Pipeline incrémental par catégorie
    @task
    def run_collect_and_process(category_params: dict) -> dict:
        """
        Collecte les nouveaux articles et traite la partition du mois courant.

        Les deux étapes sont séquentielles dans cette tâche pour conserver
        un graphe Airflow lisible (une tâche par catégorie plutôt que 2 × N).
        Une erreur marque la tâche FAILED sans affecter les autres catégories.

        L'enrichissement NLP, les tendances et la sync DuckDB sont exclus :
        ils sont délégués à dag_monthly_enrich (automatique, 1er du mois).
        """
        import logging

        log = logging.getLogger("arxiv.pipeline")

        cat_id = category_params["category_id"]
        year = category_params["current_year"]
        month = category_params["current_month"]
        month_label = f"{year}-{month:02d}"

        log.info("=" * 60)
        log.info("[Daily Watch] Début : %s | mois cible : %s", cat_id, month_label)
        log.info("=" * 60)

        # 1. Collecte : derniers N jours
        log.info("[Daily Watch] [1/2] Collecte : %s", cat_id)
        result_collect = task_collect(
            category_id=cat_id,
            lookback_days=category_params["lookback_days"],
            start_date=None,  # Toujours en lookback pour la collecte quotidienne
            end_date=None,
            max_results=category_params["max_results"],
            sort_by=category_params["sort_by"],
            sort_order=category_params["sort_order"],
            fetch_citations=category_params["fetch_citations"],
        )

        # 2. Traitement : mois courant uniquement
        log.info("[Daily Watch] [2/2] Traitement : %s / %s", cat_id, month_label)
        result_process = task_process(
            category_id=cat_id,
            year=year,
            month=month,
        )

        log.info("[Daily Watch] Terminé : %s", cat_id)
        return {
            "category_id": cat_id,
            "month": month_label,
            "collect": result_collect,
            "process": result_process,
        }

    # Chaîne d'exécution
    categories = get_active_categories()
    # expand() crée une instance de tâche par catégorie : toutes en parallèle
    run_collect_and_process.expand(category_params=categories)


daily_watch_dag()
