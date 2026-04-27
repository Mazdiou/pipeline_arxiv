"""
Schéma PyArrow de la zone GOLD et fonctions de validation.
Toutes les colonnes CLEAN sont préservées telles quelles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa


# Schéma complet de la zone GOLD
GOLD_SCHEMA = pa.schema(
    [
        # Colonnes CLEAN préservées
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("arxiv_id_short", pa.string(), nullable=False),
        pa.field("entry_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("abstract", pa.string(), nullable=False),
        pa.field("published", pa.timestamp("s", tz="UTC"), nullable=True),
        pa.field("updated", pa.timestamp("s", tz="UTC"), nullable=True),
        pa.field("published_date", pa.date32(), nullable=True),
        pa.field("published_month", pa.string(), nullable=True),
        pa.field("published_year", pa.int32(), nullable=True),
        pa.field("authors", pa.list_(pa.string()), nullable=False),
        pa.field("first_author", pa.string(), nullable=True),
        pa.field("authors_count", pa.int32(), nullable=False),
        pa.field("primary_category", pa.string(), nullable=True),
        pa.field("categories", pa.list_(pa.string()), nullable=False),
        pa.field("pdf_url", pa.string(), nullable=True),
        pa.field("doi", pa.string(), nullable=True),
        pa.field("journal_ref", pa.string(), nullable=True),
        pa.field("comment", pa.string(), nullable=True),
        # Citations Semantic Scholar (None si fetch_citations=False)
        pa.field("citation_count", pa.int32(), nullable=True),
        pa.field("reference_count", pa.int32(), nullable=True),
        pa.field("semantic_scholar_url", pa.string(), nullable=True),
        pa.field("_collected_at", pa.string(), nullable=False),
        pa.field("_processed_at", pa.string(), nullable=False),
        pa.field("_partition_category", pa.string(), nullable=False),
        pa.field("_partition_year", pa.int32(), nullable=False),
        pa.field("_partition_month", pa.int8(), nullable=False),
        # Langue
        pa.field("lang", pa.string(), nullable=False),
        pa.field("lang_confidence", pa.float32(), nullable=False),
        # Topic Modeling
        pa.field("topic_id", pa.int32(), nullable=False),
        pa.field("topic_label", pa.string(), nullable=False),
        pa.field("topic_prob", pa.float32(), nullable=False),
        pa.field("is_topic_outlier", pa.bool_(), nullable=False),
        # LaTeX
        pa.field("latex_density", pa.float32(), nullable=False),
        # Mots-clés KeyBERT
        pa.field(
            "keywords",
            pa.list_(
                pa.struct(
                    [
                        pa.field("keyword", pa.string()),
                        pa.field("score", pa.float32()),
                    ]
                )
            ),
            nullable=False,
        ),
        # Entités nommées (NER)
        pa.field(
            "entity_list",
            pa.list_(
                pa.struct(
                    [
                        pa.field("text", pa.string()),
                        pa.field("label", pa.string()),
                        pa.field("count", pa.int32()),
                    ]
                )
            ),
            nullable=False,
        ),
        # Novelty / Similarité
        pa.field("intra_similarity", pa.float32(), nullable=False),
        pa.field("novelty_score", pa.float32(), nullable=False),
        pa.field("centroid_distance", pa.float32(), nullable=False),
        # Qualité NLP
        pa.field("flesch_kincaid_grade", pa.float32(), nullable=False),
        pa.field("is_short_abstract", pa.bool_(), nullable=False),
        # Embeddings (absent si store_in_gold=False)
        pa.field("embedding", pa.list_(pa.float32()), nullable=True),
    ]
)

# Colonnes NLP obligatoires : utilisées pour la validation rapide
NLP_REQUIRED_COLUMNS: list[str] = [
    "lang",
    "lang_confidence",
    "topic_id",
    "topic_label",
    "topic_prob",
    "is_topic_outlier",
    "latex_density",
    "keywords",
    "entity_list",
    "intra_similarity",
    "novelty_score",
    "centroid_distance",
    "flesch_kincaid_grade",
    "is_short_abstract",
]

# Colonnes booléennes : doivent être sans NaN
_BOOL_COLUMNS: set[str] = {"is_topic_outlier", "is_short_abstract"}

# Colonnes numériques : doivent être sans NaN et sans infinis
_NUMERIC_COLUMNS: set[str] = {
    "lang_confidence",
    "topic_id",
    "topic_prob",
    "intra_similarity",
    "novelty_score",
    "centroid_distance",
    "flesch_kincaid_grade",
    "latex_density",
}


# Colonnes ordonnées du schema
GOLD_SCHEMA_COLUMNS: list[str] = [field.name for field in GOLD_SCHEMA]


def prepare_gold_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les conversions de types nécessaires avant écriture Parquet GOLD.
    """
    df = df.copy()

    # Timestamps UTC
    for col in ("published", "updated"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Date ISO
    if "published_date" in df.columns:
        parsed = pd.to_datetime(df["published_date"], errors="coerce")
        df["published_date"] = parsed.dt.date.where(parsed.notna(), other=None)  # type: ignore

    # Entiers
    for col in (
        "published_year",
        "authors_count",
        "citation_count",
        "reference_count",
        "_partition_year",
        "_partition_month",
        "topic_id",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Floats
    for col in (
        "lang_confidence",
        "topic_prob",
        "intra_similarity",
        "novelty_score",
        "centroid_distance",
        "flesch_kincaid_grade",
        "latex_density",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Booleans : NaN : False
    for col in ("is_topic_outlier", "is_short_abstract"):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype("bool")

    # keywords : list[{"keyword": str, "score": float}]
    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].apply(
            lambda pairs: (
                [{"keyword": kw, "score": float(sc)} for kw, sc in pairs]
                if isinstance(pairs, list)
                else []
            )
        )

    return df


def validate_gold_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Valide qu'un DataFrame contient toutes les colonnes NLP GOLD requises
    avec des valeurs cohérentes.
    """
    errors: list[str] = []

    # 1. Colonnes manquantes
    missing = [col for col in NLP_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Colonnes NLP manquantes : {missing}")
        return errors  # Arrêt précoce si colonnes absentes

    # 2. Colonnes bool sans NaN
    for col in _BOOL_COLUMNS:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Colonne '{col}' contient des NaN (utiliser fillna(False)).")

    # 3. Colonnes numériques sans NaN ni infinis
    for col in _NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            errors.append(f"Colonne '{col}' contient des NaN.")
        if np.isinf(series).any():
            errors.append(f"Colonne '{col}' contient des valeurs infinies.")

    # 4. topic_prob dans [0.0, 1.0]
    if "topic_prob" in df.columns:
        topic_prob_values = pd.to_numeric(df["topic_prob"], errors="coerce")
        if ((topic_prob_values < 0.0) | (topic_prob_values > 1.0)).any():
            errors.append("topic_prob contient des valeurs hors [0.0, 1.0].")

    # 5. Cohérence topic_id / is_topic_outlier
    if "topic_id" in df.columns and "is_topic_outlier" in df.columns:
        outlier_flag = df["is_topic_outlier"].fillna(False).infer_objects(copy=False)
        outlier_id = df["topic_id"] == -1
        if not (outlier_flag == outlier_id).all():
            errors.append(
                "Incohérence : is_topic_outlier ne correspond pas à (topic_id == -1)."
            )

    # 6. Similarités cosinus dans [0.0, 1.0]
    for col in ("intra_similarity", "novelty_score"):
        if col not in df.columns:
            continue
        col_values = pd.to_numeric(df[col], errors="coerce")
        if ((col_values < 0.0) | (col_values > 1.0)).any():
            errors.append(f"'{col}' contient des valeurs hors [0.0, 1.0].")

    # centroid_distance = distance cosinus : bornée à >= 0.0 uniquement
    if "centroid_distance" in df.columns:
        centroid_values = pd.to_numeric(df["centroid_distance"], errors="coerce")
        if (centroid_values < 0.0).any():
            errors.append("'centroid_distance' contient des valeurs négatives.")

    # 7. latex_density dans [0.0, 1.0]
    if "latex_density" in df.columns:
        latex_density_values = pd.to_numeric(df["latex_density"], errors="coerce")
        if ((latex_density_values < 0.0) | (latex_density_values > 1.0)).any():
            errors.append("'latex_density' contient des valeurs hors [0.0, 1.0].")

    # 8. lang_confidence dans [0.0, 1.0]
    if "lang_confidence" in df.columns:
        lang_confidence_values = pd.to_numeric(df["lang_confidence"], errors="coerce")
        if ((lang_confidence_values < 0.0) | (lang_confidence_values > 1.0)).any():
            errors.append("'lang_confidence' contient des valeurs hors [0.0, 1.0].")

    return errors
