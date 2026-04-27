"""
dashboard/core/airflow_client.py

Client HTTP pour l'API REST Airflow.

Permet au dashboard de déclencher les DAGs sans dépendance directe
sur les bibliothèques Airflow (qui requièrent Linux/WSL2).

Variables d'environnement requises :
    AIRFLOW_URL      : URL de base de l'API Airflow  (ex: http://localhost:8080)
    AIRFLOW_USERNAME : Identifiant Airflow            (défaut : admin)
    AIRFLOW_PASSWORD : Mot de passe Airflow           (défaut : admin)

DAGs supportés :
    arxiv_bootstrap  — bootstrap initial d'une catégorie
    arxiv_daily_watch — watch quotidien (déclenché manuellement si besoin)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────
DAG_BOOTSTRAP   = "arxiv_bootstrap"
DAG_DAILY_WATCH = "arxiv_daily_watch"

_API_VERSION = "v1"


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _airflow_base_url() -> str:
    """Retourne l'URL de base de l'API Airflow sans slash final."""
    return os.getenv("AIRFLOW_URL", "http://localhost:8080").rstrip("/")


def _airflow_auth() -> tuple[str, str]:
    """Retourne (username, password) pour l'authentification Basic Auth."""
    return (
        os.getenv("AIRFLOW_USERNAME", "admin"),
        os.getenv("AIRFLOW_PASSWORD", "admin"),
    )


def _post_dag_run(dag_id: str, conf: dict) -> tuple[bool, str]:
    """
    Déclenche un DAG via l'API REST Airflow.

    Args:
        dag_id : Identifiant du DAG (ex: 'arxiv_bootstrap').
        conf   : Dictionnaire de configuration passé au DAG.

    Returns:
        Tuple (succès: bool, message: str).
    """
    try:
        import requests
    except ImportError:
        return False, "La bibliothèque 'requests' est manquante (pip install requests)."

    url  = f"{_airflow_base_url()}/api/{_API_VERSION}/dags/{dag_id}/dagRuns"
    auth = _airflow_auth()

    try:
        response = requests.post(
            url,
            json={"conf": conf},
            auth=auth,
            timeout=10,
            headers={"Content-Type": "application/json"},
        )
    except requests.ConnectionError:
        msg = (
            f"Impossible de joindre Airflow à '{_airflow_base_url()}'. "
            "Vérifiez que le serveur est démarré (airflow webserver)."
        )
        logger.error("trigger_dag '%s' : %s", dag_id, msg)
        return False, msg
    except requests.Timeout:
        msg = f"Délai d'attente dépassé pour l'API Airflow ({_airflow_base_url()})."
        logger.error("trigger_dag '%s' : %s", dag_id, msg)
        return False, msg
    except Exception as exc:
        msg = f"Erreur inattendue : {exc}"
        logger.error("trigger_dag '%s' : %s", dag_id, msg)
        return False, msg

    if response.status_code == 200:
        run_id = response.json().get("dag_run_id", "—")
        msg = f"DAG run lancé (run_id: {run_id})."
        logger.info("trigger_dag '%s' : %s", dag_id, msg)
        return True, msg

    if response.status_code == 401:
        return False, "Authentification Airflow échouée. Vérifiez AIRFLOW_USERNAME et AIRFLOW_PASSWORD."

    if response.status_code == 404:
        return False, f"DAG '{dag_id}' introuvable dans Airflow. Le DAG est-il activé ?"

    if response.status_code == 409:
        return False, "Un run de ce DAG est déjà en cours."

    # Erreur générique avec message Airflow
    try:
        detail = response.json().get("detail", response.text[:200])
    except Exception:
        detail = response.text[:200]
    return False, f"Airflow a retourné HTTP {response.status_code} : {detail}"


# =========================================================================== #
# API publique                                                                 #
# =========================================================================== #

def trigger_bootstrap(category_id: str) -> tuple[bool, str]:
    """
    Déclenche le DAG arxiv_bootstrap pour une collecte initiale complète.

    Args:
        category_id : Identifiant interne de la catégorie (ex: 'stat_ML').

    Returns:
        (succès: bool, message: str)
    """
    return _post_dag_run(
        dag_id=DAG_BOOTSTRAP,
        conf={"category_id": category_id},
    )


def trigger_reenrich(category_id: str) -> tuple[bool, str]:
    """
    Déclenche le DAG arxiv_bootstrap en sautant la collecte et le traitement.
    Utile pour re-lancer uniquement l'enrichissement NLP + tendances + sync.

    Args:
        category_id : Identifiant interne de la catégorie (ex: 'stat_ML').

    Returns:
        (succès: bool, message: str)
    """
    return _post_dag_run(
        dag_id=DAG_BOOTSTRAP,
        conf={
            "category_id":  category_id,
            "skip_collect": True,
            "skip_process": True,
        },
    )


def trigger_daily_watch() -> tuple[bool, str]:
    """
    Déclenche le DAG arxiv_daily_watch manuellement (hors planification).

    Returns:
        (succès: bool, message: str)
    """
    return _post_dag_run(dag_id=DAG_DAILY_WATCH, conf={})


def is_airflow_available() -> bool:
    """
    Vérifie rapidement que l'API Airflow est joignable.

    Returns:
        True si l'API répond, False sinon.
    """
    try:
        import requests
        url  = f"{_airflow_base_url()}/api/{_API_VERSION}/health"
        auth = _airflow_auth()
        resp = requests.get(url, auth=auth, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_recent_runs(limit: int = 20) -> list[dict]:
    """
    Retourne les derniers runs de arxiv_bootstrap et arxiv_daily_watch,
    enrichis avec la catégorie extraite du conf Airflow.

    Returns:
        Liste de dicts triée par start_date décroissant :
        {dag_id, category, state, start_date, end_date, run_id}
        Liste vide si Airflow est indisponible.
    """
    try:
        import requests
        auth = _airflow_auth()
        base = _airflow_base_url()
        results: list[dict] = []

        for dag_id in (DAG_BOOTSTRAP, DAG_DAILY_WATCH):
            url = f"{base}/api/{_API_VERSION}/dags/{dag_id}/dagRuns"
            resp = requests.get(
                url,
                auth=auth,
                params={"limit": limit, "order_by": "-start_date"},
                timeout=5,
            )
            if resp.status_code != 200:
                continue

            for r in resp.json().get("dag_runs", []):
                conf = r.get("conf") or {}
                category = conf.get("category_id", "—")
                results.append({
                    "dag_id":     dag_id,
                    "category":   category,
                    "state":      r.get("state", "unknown"),
                    "start_date": r.get("start_date"),
                    "end_date":   r.get("end_date"),
                    "run_id":     r.get("dag_run_id", ""),
                })

        results.sort(key=lambda r: r["start_date"] or "", reverse=True)
        return results[:limit]
    except Exception as exc:
        logger.debug("get_recent_runs : %s", exc)
        return []


def get_dag_status(dag_id: str, limit: int = 5) -> list[dict]:
    """
    Retourne les derniers runs d'un DAG.

    Args:
        dag_id : Identifiant du DAG.
        limit  : Nombre de runs à retourner (défaut : 5).

    Returns:
        Liste de dicts {dag_run_id, state, start_date, end_date}.
        Liste vide si Airflow est indisponible.
    """
    try:
        import requests
        url  = f"{_airflow_base_url()}/api/{_API_VERSION}/dags/{dag_id}/dagRuns"
        auth = _airflow_auth()
        resp = requests.get(
            url,
            auth=auth,
            params={"limit": limit, "order_by": "-start_date"},
            timeout=5,
        )
        if resp.status_code != 200:
            return []
        runs = resp.json().get("dag_runs", [])
        return [
            {
                "dag_run_id": r.get("dag_run_id"),
                "state":      r.get("state"),
                "start_date": r.get("start_date"),
                "end_date":   r.get("end_date"),
            }
            for r in runs
        ]
    except Exception as exc:
        logger.debug("get_dag_status '%s' : %s", dag_id, exc)
        return []
