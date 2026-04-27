"""
Utilitaires de gestion des dates pour la collecte arXiv.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.settings import CollectorConfig


# Constantes
DATE_FORMAT = "%Y-%m-%d"
ARXIV_DT_FORMAT = "%Y%m%d%H%M%S"


def resolve_date_window(config: "CollectorConfig") -> tuple[datetime, datetime]:
    """
    Calcule la fenêtre temporelle (start_dt, end_dt).

    Priorité :
        1. start_date + end_date explicites
        2. lookback_days depuis aujourd'hui

    Returns:
        (start_dt, end_dt) en datetime UTC-naïf
    """

    # Détermination de la date de fin
    if config.end_date:
        end_dt = _parse_date(config.end_date).replace(hour=23, minute=59, second=59)
    else:
        end_dt = _now_utc()

    # Détermination de la date de début
    if config.start_date:
        start_dt = _parse_date(config.start_date)
    else:
        start_dt = end_dt - timedelta(days=config.lookback_days)

    # Validation
    if start_dt >= end_dt:
        raise ValueError(
            f"La date de début ({start_dt.date()}) doit être "
            f"strictement antérieure à la date de fin ({end_dt.date()})."
        )

    return start_dt, end_dt


def format_arxiv_date_filter(start_dt: datetime, end_dt: datetime) -> str:
    """
    Formate un filtre temporel compatible avec l'API arXiv.

    Format :
        submittedDate:[YYYYMMDDHHMMSS TO YYYYMMDDHHMMSS]
    """

    start = start_dt.strftime(ARXIV_DT_FORMAT)
    end = end_dt.strftime(ARXIV_DT_FORMAT)

    return f"submittedDate:[{start} TO {end}]"


def is_within_window(dt: datetime, start_dt: datetime, end_dt: datetime) -> bool:
    """
    Vérifie si une datetime appartient à la fenêtre [start_dt, end_dt].
    """
    dt_naive = dt.replace(tzinfo=None)
    return start_dt <= dt_naive <= end_dt


# Helpers privés
def _parse_date(date_str: str) -> datetime:
    """Parse une date 'YYYY-MM-DD'."""
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except ValueError as e:
        raise ValueError(
            f"Format de date invalide : '{date_str}'. Attendu : YYYY-MM-DD"
        ) from e


def _now_utc() -> datetime:
    """Retourne la datetime courante UTC-naïf."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
