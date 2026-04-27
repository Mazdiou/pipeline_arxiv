"""
Loader pour fact_articles à partir des partitions GOLD.
"""

from __future__ import annotations

import json
import logging

import duckdb
import numpy as np
import pandas as pd

from ..reader import BaseDataReader

logger = logging.getLogger(__name__)

# Colonnes GOLD conservées dans fact_articles
_GOLD_COLUMNS = [
    "arxiv_id",
    "title",
    "abstract",
    "first_author",
    "authors_count",
    "published_date",
    "updated",
    "published_month",
    "pdf_url",
    "doi",
    "journal_ref",
    "citation_count",
    "reference_count",
    "semantic_scholar_url",
    "primary_category",
    "topic_id",
    "topic_label",
    "topic_prob",
    "is_topic_outlier",
    "lang",
    "lang_confidence",
    "novelty_score",
    "intra_similarity",
    "centroid_distance",
    "latex_density",
    "flesch_kincaid_grade",
    "is_short_abstract",
    "keywords",
    "entity_list",
]

# Ordre canonique des colonnes pour l'insertion (correspond au DDL de schema.py)
_SCHEMA_COLUMNS = [
    "arxiv_id",
    "title",
    "abstract",
    "first_author",
    "authors_count",
    "published_date",
    "updated",
    "published_month",
    "pdf_url",
    "doi",
    "journal_ref",
    "citation_count",
    "reference_count",
    "semantic_scholar_url",
    "category",
    "primary_category",
    "topic_id",
    "topic_label",
    "topic_prob",
    "is_topic_outlier",
    "lang",
    "lang_confidence",
    "novelty_score",
    "intra_similarity",
    "centroid_distance",
    "latex_density",
    "flesch_kincaid_grade",
    "is_short_abstract",
    "keywords",
    "entity_list",
]


def load_articles(
    conn: duckdb.DuckDBPyConnection,
    category: str,
    reader: BaseDataReader,
) -> int:
    """
    Lit toutes les partitions GOLD d'une catégorie et insère dans fact_articles.

    Suppose que delete_category() a déjà été appelé pour cette catégorie.

    Args:
        conn     : Connexion DuckDB ouverte.
        category : Catégorie arXiv (ex: 'stat_ML').
        reader   : Reader abstrait (LocalReader ou MinioReader).

    Returns:
        Nombre total d'articles insérés.
    """
    partitions = reader.read_gold_partitions(category)
    if not partitions:
        logger.warning("Aucune partition GOLD trouvée pour '%s'.", category)
        return 0

    total_inserted = 0
    for raw_df in partitions:
        df = _prepare_gold_partition(raw_df, category)
        if df.empty:
            continue
        col_list = ", ".join(df.columns.tolist())
        conn.register("_articles_batch", df)
        conn.execute(
            f"INSERT INTO fact_articles ({col_list}) SELECT * FROM _articles_batch"
        )
        conn.unregister("_articles_batch")
        total_inserted += len(df)
        logger.debug("  partition → %d articles chargés.", len(df))

    logger.info(
        "fact_articles '%s' : %d articles insérés depuis %d partitions.",
        category,
        total_inserted,
        len(partitions),
    )
    return total_inserted


def _prepare_gold_partition(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Prépare un DataFrame GOLD brut pour insertion dans fact_articles.

    Args:
        df       : DataFrame GOLD brut lu depuis le reader.
        category : Catégorie arXiv (ajoutée comme colonne).

    Returns:
        DataFrame prêt pour insertion. Vide si sans données exploitables.
    """
    if df.empty:
        return pd.DataFrame()

    # Conserver uniquement les colonnes utiles
    available = [col for col in _GOLD_COLUMNS if col in df.columns]
    df = df[available].copy()

    # Catégorie non stockée dans le Parquet GOLD
    df["category"] = category

    # Sérialiser les colonnes structurées en JSON
    for col in ("keywords", "entity_list"):
        if col in df.columns:
            df[col] = df[col].apply(_serialize_to_json)

    # Normalisation des types
    if "topic_id" in df.columns:
        df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").astype("Int32")

    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(
            df["published_date"], errors="coerce"
        ).dt.date

    if "updated" in df.columns:
        df["updated"] = pd.to_datetime(df["updated"], errors="coerce", utc=True).dt.date

    for col in ("is_topic_outlier", "is_short_abstract"):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # Réordonner selon le schéma DuckDB
    final_cols = [c for c in _SCHEMA_COLUMNS if c in df.columns]
    return df[final_cols]


def _serialize_to_json(value) -> str | None:
    """
    Sérialise une valeur Python (list, dict, numpy.ndarray) en JSON string pour DuckDB.

    pd.read_parquet() peut retourner des numpy.ndarray d'objets Python pour les colonnes
    PyArrow list_(struct(...)). json.dumps() ne gère pas nativement les types numpy.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    except (TypeError, ValueError):
        return None


def _json_default(obj):
    """Encodeur JSON de secours pour les types numpy scalaires et arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
