"""
DAG Monthly Enrich : Enrichissement NLP, tendances et sync DuckDB.

Planification :
    Le 1er de chaque mois à 3h00 UTC : schedule="0 3 1 * *"

Comportement :
    Pour chaque catégorie configurée, en parallèle :
        a. Enrichissement NLP : CLEAN → GOLD sur le mois précédent
        b. Tendances          : GOLD  → TRENDS sur le mois précédent
        c. Sync DuckDB        : GOLD + TRENDS → dashboard.duckdb

Mois cible :
    Calculé depuis data_interval_end (date logique Airflow) pour être robuste
    au catchup et aux re-runs manuels. Toujours le mois précédant le run.
    Exemple : run du 2026-04-01 → enrichit les données de 2026-03.

Airflow Variables optionnelles :
    ARXIV_MONTHLY_SKIP_EXISTING (bool, défaut : true)
        false → force le re-enrichissement même si GOLD existe déjà.

Gestion des erreurs :
    - Les erreurs de configuration (catégorie manquante, variable invalide)
      lèvent une ValueError et marquent la tâche get_categories FAILED.
    - Les erreurs d'enrichissement NLP sont fatales : elles interrompent
      le traitement de la catégorie concernée (trends + sync non exécutés).
    - Les erreurs de tendances ou de sync DuckDB sont loggées avec contexte
      complet et remontées comme RuntimeError pour marquer la tâche FAILED.
    - Une catégorie en échec n'affecte pas les autres (isolation via expand()).
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dags.pipeline_tasks import (
    task_enrich_nlp,
    task_enrich_trends,
    task_sync_duckdb,
)


# DAG


@dag(
    dag_id="arxiv_monthly_enrich",
    description=(
        "Enrichissement NLP, tendances et sync DuckDB sur le mois précédent, "
        "pour toutes les catégories arXiv configurées."
    ),
    schedule="0 3 1 * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=2,  # NLP est coûteux : limiter les catégories simultanées
    tags=["arxiv", "enrich", "monthly"],
    params={
        "force_year": Param(
            default=0,
            type="integer",
            description=(
                "Forcer l'année cible (0 = calculée automatiquement depuis le schedule). "
                "À utiliser avec force_month pour rejouer un mois précis."
            ),
        ),
        "force_month": Param(
            default=0,
            type="integer",
            description=(
                "Forcer le mois cible 1-12 (0 = calculé automatiquement). "
                "À utiliser avec force_year pour rejouer un mois précis."
            ),
        ),
    },
    doc_md=__doc__,
)
def monthly_enrich_dag():

    import logging

    log = logging.getLogger("arxiv.pipeline")

    # Chargement des catégories + calcul du mois cible
    @task
    def get_categories_for_enrich(params: dict = None, **context) -> list[dict]:
        """
        Lit toutes les catégories configurées et calcule le mois à enrichir.

        Le mois cible est dérivé de data_interval_end (date logique Airflow)
        pour être robuste aux re-runs et au catchup. Les paramètres force_year
        et force_month permettent de surcharger ce calcul manuellement.

        Retourne une liste de dicts triée par category_id : un dict par
        catégorie : contenant les paramètres NLP/trends et le mois cible.

        Raises:
            ValueError : si la variable ARXIV_MONTHLY_SKIP_EXISTING est
                         malformée, ou si aucune catégorie n'est configurée
                         et que l'on considère cela comme une erreur de config.
        """
        import logging
        from airflow.models import Variable
        from dashboard.core.config_store import ConfigStore

        log = logging.getLogger("arxiv.pipeline")

        # Calcul du mois cible
        params = params or {}
        force_year = int(params.get("force_year", 0))
        force_month = int(params.get("force_month", 0))

        if force_year > 0 and force_month > 0:
            # Surcharge manuelle : validation basique
            if not (1 <= force_month <= 12):
                raise ValueError(
                    f"force_month={force_month} invalide : doit être compris entre 1 et 12."
                )
            if force_year < 2000 or force_year > 2100:
                raise ValueError(
                    f"force_year={force_year} invalide : valeur hors plage acceptable."
                )
            target_year, target_month = force_year, force_month
            log.info(
                "[Monthly Enrich] Mois cible forcé manuellement : %04d-%02d",
                target_year,
                target_month,
            )
        else:
            # Mois précédent depuis data_interval_end (robuste au catchup)
            logical_date: datetime = context.get("data_interval_end") or datetime.now(
                timezone.utc
            )
            if logical_date.month == 1:
                target_year, target_month = logical_date.year - 1, 12
            else:
                target_year, target_month = logical_date.year, logical_date.month - 1
            log.info(
                "[Monthly Enrich] Mois cible calculé depuis data_interval_end=%s : %04d-%02d",
                logical_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                target_year,
                target_month,
            )

        # Lecture de la variable skip_existing
        raw_skip = Variable.get("ARXIV_MONTHLY_SKIP_EXISTING", default_var="true")
        if raw_skip.lower() not in ("true", "false"):
            raise ValueError(
                f"Variable Airflow 'ARXIV_MONTHLY_SKIP_EXISTING' invalide : "
                f"valeur '{raw_skip}' inattendue. Attendu : 'true' ou 'false'."
            )
        skip_existing = raw_skip.lower() == "true"

        # Chargement des catégories
        store = ConfigStore()
        configs = store.load_all()

        if not configs:
            log.warning(
                "[Monthly Enrich] Aucune catégorie dans pipeline_configs.json : "
                "rien à faire. Configurez au moins une catégorie sur la Page 0 "
                "du dashboard avant de déclencher ce DAG."
            )
            return []

        # Tri déterministe pour des logs reproductibles d'un run à l'autre
        category_params = []
        for cat_id, cfg in sorted(configs.items()):
            category_params.append(
                {
                    "category_id": cat_id,
                    "display_name": cfg.display_name,
                    "nr_topics": cfg.nr_topics,
                    "min_months_for_trend": cfg.min_months_for_trend,
                    "z_score_window": cfg.z_score_window,
                    "semantic_drift_enabled": cfg.semantic_drift_enabled,
                    "skip_existing": skip_existing,
                    "target_year": target_year,
                    "target_month": target_month,
                }
            )

        log.info(
            "[Monthly Enrich] %d catégorie(s) à enrichir | mois cible : %04d-%02d "
            "| skip_existing=%s | ordre : %s",
            len(category_params),
            target_year,
            target_month,
            skip_existing,
            [p["category_id"] for p in category_params],
        )
        return category_params

    # Enrichissement par catégorie
    @task
    def run_enrich_and_sync(category_params: dict) -> dict:
        """
        Exécute NLP → Tendances → Sync DuckDB pour une catégorie et un mois.

        Chaque étape est exécutée séquentiellement. Une erreur à n'importe
        quelle étape interrompt le traitement de la catégorie, logue un
        contexte d'erreur complet (traceback + métriques partielles) et
        marque la tâche FAILED sans affecter les autres catégories.

        Args:
            category_params : dict produit par get_categories_for_enrich.

        Returns:
            dict avec les clés : category_id, month, nlp, trends, duckdb,
            elapsed_seconds, status.

        Raises:
            RuntimeError : propagée depuis task_enrich_nlp, task_enrich_trends
                           ou task_sync_duckdb en cas d'échec.
        """
        import logging
        import time

        log = logging.getLogger("arxiv.pipeline")

        cat_id = category_params["category_id"]
        display_name = category_params["display_name"]
        year = category_params["target_year"]
        month = category_params["target_month"]
        month_label = f"{year}-{month:02d}"

        t_start = time.monotonic()

        # Résultats partiels : remplis au fur et à mesure pour le logging
        # d'erreur même si une étape échoue
        result: dict = {
            "category_id": cat_id,
            "month": month_label,
            "nlp": None,
            "trends": None,
            "duckdb": None,
            "status": "started",
        }

        log.info(
            "[Monthly Enrich] Début   : %s (%s) | mois cible : %s",
            cat_id,
            display_name,
            month_label,
        )
        log.info(
            "[Monthly Enrich] Config  : nr_topics=%s | z_score_window=%s "
            "| min_months=%s | semantic_drift=%s | skip_existing=%s",
            category_params["nr_topics"],
            category_params["z_score_window"],
            category_params["min_months_for_trend"],
            category_params["semantic_drift_enabled"],
            category_params["skip_existing"],
        )

        # Étape 1 : Enrichissement NLP
        t0 = time.monotonic()
        log.info("[Monthly Enrich] [1/3] NLP : %s / %s", cat_id, month_label)
        try:
            result_nlp = task_enrich_nlp(
                category_id=cat_id,
                nr_topics=category_params["nr_topics"],
                year=year,
                month=month,
                skip_existing=category_params["skip_existing"],
            )
            result["nlp"] = result_nlp
            log.info(
                "[Monthly Enrich] [1/3] NLP OK : %s / %s : %s : %.1fs",
                cat_id,
                month_label,
                result_nlp,
                time.monotonic() - t0,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            result["status"] = "failed_nlp"
            log.error(
                "[Monthly Enrich] [1/3] NLP ÉCHOUÉ : %s / %s\n"
                "  Erreur    : %s\n"
                "  Résultats partiels : %s\n"
                "  Temps écoulé : %.1fs\n"
                "%s",
                cat_id,
                month_label,
                exc,
                result,
                elapsed,
                traceback.format_exc(),
            )
            # Erreur NLP = fatale pour cette catégorie : trends et sync
            # dépendent de GOLD qui n'existe pas/est incomplet
            raise RuntimeError(f"[NLP] {cat_id}/{month_label} : {exc}") from exc

        # Étape 2 : Détection de tendances
        t0 = time.monotonic()
        log.info("[Monthly Enrich] [2/3] Tendances : %s / %s", cat_id, month_label)
        try:
            result_trends = task_enrich_trends(
                category_id=cat_id,
                semantic_drift_enabled=category_params["semantic_drift_enabled"],
                min_months_for_trend=category_params["min_months_for_trend"],
                z_score_window=category_params["z_score_window"],
                year=year,
                month=month,
            )
            result["trends"] = result_trends
            log.info(
                "[Monthly Enrich] [2/3] Tendances OK : %s / %s : %s : %.1fs",
                cat_id,
                month_label,
                result_trends,
                time.monotonic() - t0,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            result["status"] = "failed_trends"
            log.error(
                "[Monthly Enrich] [2/3] Tendances ÉCHOUÉES : %s / %s\n"
                "  Erreur    : %s\n"
                "  Résultats partiels : %s\n"
                "  Temps écoulé : %.1fs\n"
                "%s",
                cat_id,
                month_label,
                exc,
                result,
                elapsed,
                traceback.format_exc(),
            )
            raise RuntimeError(f"[Tendances] {cat_id}/{month_label} : {exc}") from exc

        # Étape 3 : Synchronisation DuckDB
        t0 = time.monotonic()
        log.info("[Monthly Enrich] [3/3] Sync DuckDB : %s", cat_id)
        try:
            result_dw = task_sync_duckdb(
                category_id=cat_id,
                display_name=display_name,
            )
            result["duckdb"] = result_dw
            log.info(
                "[Monthly Enrich] [3/3] Sync DuckDB OK : %s : %s : %.1fs",
                cat_id,
                result_dw,
                time.monotonic() - t0,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            result["status"] = "failed_duckdb"
            log.error(
                "[Monthly Enrich] [3/3] Sync DuckDB ÉCHOUÉE : %s\n"
                "  Erreur    : %s\n"
                "  Résultats partiels : %s\n"
                "  Temps écoulé : %.1fs\n"
                "%s",
                cat_id,
                exc,
                result,
                elapsed,
                traceback.format_exc(),
            )
            raise RuntimeError(f"[Sync DuckDB] {cat_id} : {exc}") from exc

        # Résumé final
        elapsed_total = time.monotonic() - t_start
        result["elapsed_seconds"] = round(elapsed_total, 1)
        result["status"] = "success"

        log.info(
            "[Monthly Enrich] ✓ Terminé : %s / %s : %.1fs total\n"
            "  NLP    : %s\n"
            "  Trends : %s\n"
            "  DuckDB : %s",
            cat_id,
            month_label,
            elapsed_total,
            result["nlp"],
            result["trends"],
            result["duckdb"],
        )

        return result

    # Chaîne d'exécution
    categories = get_categories_for_enrich()
    run_enrich_and_sync.expand(category_params=categories)


monthly_enrich_dag()
