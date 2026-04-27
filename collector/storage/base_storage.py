"""
Interface abstraite et helpers partagés entre RawStorage et MinioStorage.

Les deux backends sont des implémentations sœurs de BaseStorage :
aucune ne doit dépendre de l'autre.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime


# Constantes internes
_VERSION_RE = re.compile(r"v(\d+)$")


# Interface abstraite
class BaseStorage(ABC):
    """Interface commune pour tous les backends de stockage RAW."""

    @abstractmethod
    def save(self, articles: list[dict], category: str, run_date: datetime) -> str:
        """
        Sauvegarde les articles et retourne le chemin/clé du fichier.

        Args:
            articles : Articles sérialisés à stocker
            category : Catégorie arXiv
            run_date : Date du run

        Returns:
            Chemin ou clé du fichier stocké
        """
        ...

    @abstractmethod
    def load(self, path: str) -> list[dict]:
        """
        Charge les articles depuis un fichier ou une clé existante.

        Args:
            path : Chemin local ou clé objet

        Returns:
            Liste d'articles sérialisés
        """
        ...


# Helpers partagés
def _safe_category_name(category: str) -> str:
    """
    Normalise un nom de catégorie arXiv en nom sûr pour filesystem / S3.

    Remplace les séparateurs incompatibles ('.' et '/') par des underscores.
    """
    safe = category.replace(".", "_").replace("/", "_")
    return safe or "unknown"


def _extract_version(arxiv_id: str) -> int:
    """
    Extrait le numéro de version d'un identifiant arXiv.

    Exemple :
        2401.12345v3 → 3
    """
    match = _VERSION_RE.search(arxiv_id)
    return int(match.group(1)) if match else 1


def _group_by_month(
    articles: list[dict],
    fallback_date: datetime,
    logger: logging.Logger | None = None,
) -> dict[datetime, list[dict]]:
    """
    Regroupe les articles par mois de publication.

    Le champ attendu est : 'published_month' (format YYYY-MM).

    Utilisée par RawStorage et MinioStorage pour garantir un
    comportement identique.
    """
    groups: dict[datetime, list[dict]] = {}
    fallback_month = fallback_date.replace(day=1)

    for article in articles:
        month_str = article.get("published_month")  # ex: "2024-03"

        if month_str:
            try:
                month_dt = datetime.strptime(month_str, "%Y-%m")
            except ValueError:
                month_dt = fallback_month
        else:
            month_dt = fallback_month

        groups.setdefault(month_dt, []).append(article)

    if logger:
        summary = [
            (month.strftime("%Y-%m"), len(items)) for month, items in groups.items()
        ]
        logger.debug(f"Articles regroupés par mois : {summary}")

    return groups


def _merge(
    existing: list[dict],
    incoming: list[dict],
    logger: logging.Logger | None = None,
) -> tuple[list[dict], int, int, int]:
    """
    Fusionne deux listes d'articles avec déduplication intelligente.

    Stratégie :
        - Clé de déduplication = arxiv_id_short (sans suffixe de version)
        - Si une version plus récente arrive, elle remplace l'ancienne

    Utilisée par RawStorage et MinioStorage pour garantir un comportement identique.
    """

    # Index des articles existants par id court
    existing_index: dict[str, dict] = {
        a.get("arxiv_id_short", a["arxiv_id"]): a for a in existing
    }

    new_count = 0
    duplicate_count = 0
    updated_count = 0

    for article in incoming:
        short_id = article.get("arxiv_id_short", article["arxiv_id"])

        if short_id not in existing_index:
            existing_index[short_id] = article
            new_count += 1
            continue

        existing_article = existing_index[short_id]

        existing_ver = _extract_version(existing_article.get("arxiv_id", ""))
        incoming_ver = _extract_version(article.get("arxiv_id", ""))

        if incoming_ver > existing_ver:
            existing_index[short_id] = article
            updated_count += 1

            if logger:
                logger.debug(
                    f"Version mise à jour : "
                    f"{short_id} v{existing_ver} → v{incoming_ver}"
                )
        else:
            duplicate_count += 1

    merged_articles = list(existing_index.values())

    return merged_articles, new_count, duplicate_count, updated_count
