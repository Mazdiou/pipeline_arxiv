"""
Génération d'identifiants uniques de run.
"""

from __future__ import annotations

from datetime import datetime, timezone


def build_run_id(key: str = "") -> str:
    """
    Génère un identifiant unique de run basé sur l'horodatage UTC.

    Utilisé pour nommer le fichier de log ET pour l'entrée dans run_history.json,
    garantissant que les deux sont liés au même run.

    Exemples :
        build_run_id("cs.AI")          → "20260222_131406_cs_AI"
        build_run_id("collect_cs_AI")  → "20260222_131406_collect_cs_AI"
    """
    ts = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    safe = (key or "unknown").replace(".", "_").replace("/", "_")
    return f"{ts}_{safe}"
