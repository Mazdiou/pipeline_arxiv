"""
Construction de la query arXiv à partir des paramètres de configuration.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from .date import format_arxiv_date_filter

if TYPE_CHECKING:
    from ..config.settings import CollectorConfig


def build_query(config: "CollectorConfig", start_dt: datetime, end_dt: datetime) -> str:
    """
    Construit la query complète pour l'API arXiv.

    Combine dans l'ordre :
        1. Filtre de catégorie  (cat:)
        2. Filtre de mots-clés  (all:)
        3. Filtre temporel      (submittedDate:[...])
    """
    parts: list[str] = []

    # 1. Filtre catégorie
    if config.category:
        parts.append(_build_category_filter(config.category))

    # 2. Filtre mots-clés
    if config.query:
        parts.append(_build_keyword_filter(config.query))

    # 3. Filtre temporel
    parts.append(format_arxiv_date_filter(start_dt, end_dt))

    return " AND ".join(parts)


#  Builders de sous-filtres (extensibles)
def _build_category_filter(category: str) -> str:
    """
    Construit le filtre de catégorie.
    """
    return f"cat:{category.strip()}"


def _build_keyword_filter(query: str, field: str = "all") -> str:
    """
    Construit le filtre de mots-clés.

    Args:
        query : Termes de recherche (peut contenir plusieurs mots)
        field : Champ cible — 'all', 'ti' (titre), 'abs' (abstract), 'au' (auteur)
    """
    clean = query.strip()
    # Encapsulation pour les expressions multi-mots
    if " " in clean and not (clean.startswith('"') and clean.endswith('"')):
        clean = f'"{clean}"'
    return f"{field}:{clean}"
