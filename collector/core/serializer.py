"""
Transformation d'un objet arxiv.Result en JSON-sérialisable.

Contrat de sortie :
    Le dictionnaire retourné DOIT toujours contenir les champs de REQUIRED_FIELDS.
"""

from __future__ import annotations
import re
from datetime import datetime, timezone

import arxiv

from ..config.constants import REQUIRED_FIELDS


def serialize_article(result: arxiv.Result) -> dict:
    """
    Convertit un arxiv.Result en dictionnaire JSON-sérialisable.

    La structure de sortie est organisée en sections logiques :
        - Identifiants   : arxiv_id, arxiv_id_short, entry_id
        - Contenu        : title, abstract
        - Dates          : published, updated, published_date, published_month, published_year
        - Auteurs        : authors, first_author, authors_count
        - Catégories     : primary_category, categories
        - Liens          : pdf_url, doi, journal_ref, comment
        - Citations      : citation_count, reference_count
                           (None par défaut — renseigné si fetch_citations=True dans CollectorConfig)
        - Traçabilité    : _collected_at

    Args:
        result : Objet retourné par l'itérateur arxiv.Client.results().

    Raises:
        ValueError : si un champ requis est absent ou None.
    """
    pub = result.published  # datetime
    upd = result.updated  # datetime | None
    auths = result.authors  # list[arxiv.Author]

    # Identifiant complet avec suffixe de version (ex: "2301.00001v2")
    _full_id = result.entry_id.split("/abs/")[-1]

    # Clé de déduplication : suffixe vN supprimé (ex: "2301.00001")
    _short_id = re.sub(r"v\d+$", "", _full_id)

    article: dict = {
        # Identifiants
        # Inclut le suffixe de version pour la traçabilité complète
        "arxiv_id": _full_id,
        # Sans suffixe de version = clé de déduplication
        "arxiv_id_short": _short_id,
        # URL complète de la fiche arXiv
        "entry_id": result.entry_id,
        # Contenu textuel
        "title": result.title,
        "abstract": result.summary,
        # Dates
        # ISO 8601 complet avec timezone (ex: "2024-03-15T14:32:00+00:00")
        "published": pub.isoformat(),
        "updated": upd.isoformat() if upd else None,
        # Format YYYY-MM-DD (ex: "2024-03-15")
        "published_date": pub.strftime("%Y-%m-%d"),
        # Clé d'agrégation mensuelle (ex: "2024-03")
        "published_month": pub.strftime("%Y-%m"),
        # Année seule (ex: 2024)
        "published_year": pub.year,
        # Auteurs
        # Liste complète des noms (prénom + nom)
        "authors": [a.name for a in auths],
        # Premier auteur seul
        "first_author": auths[0].name,
        # Nombre total d'auteurs
        "authors_count": len(auths),
        # Catégories arXiv
        # Catégorie principale déclarée par les auteurs
        "primary_category": result.primary_category,
        # Toutes les catégories (un article peut en avoir plusieurs)
        "categories": result.categories,
        # Liens externes
        "pdf_url": result.pdf_url,
        "doi": result.doi,
        "journal_ref": result.journal_ref,
        # Note libre des auteurs (ex: "Accepted at NeurIPS 2024, 12 pages")
        "comment": result.comment,
        # Citations Semantic Scholar
        # None par défaut — renseigné après collecte si fetch_citations=True
        "citation_count": None,
        "reference_count": None,
        "semantic_scholar_url": None,
        # Traçabilité de collecte
        # Timestamp UTC de collecte (ISO 8601 avec offset +00:00)
        "_collected_at": datetime.now(timezone.utc).isoformat(),
    }

    # Validation du contrat de sortie
    _validate_required_fields(article)

    return article


def _validate_required_fields(article: dict) -> None:
    """
    Vérifie que tous les champs de REQUIRED_FIELDS sont présents et non-None.

    Args:
        article : Dictionnaire produit par serialize_article().

    Raises:
        ValueError : si un champ requis est absent ou vaut None.
    """
    missing_or_none = [field for field in REQUIRED_FIELDS if article.get(field) is None]
    if missing_or_none:
        raise ValueError(
            f"Champs requis absents ou None dans l'article sérialisé : "
            f"{sorted(missing_or_none)}. "
            f"Vérifiez que l'objet arxiv.Result est complet."
        )
