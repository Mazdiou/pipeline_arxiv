"""
Écriture des DataFrames nettoyés en format Parquet dans la zone CLEAN.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Schéma Parquet CLEAN
CLEAN_SCHEMA = pa.schema(
    [
        # Identifiants
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("arxiv_id_short", pa.string(), nullable=False),
        pa.field("entry_id", pa.string(), nullable=False),
        # Contenu textuel
        pa.field("title", pa.string(), nullable=False),
        pa.field("abstract", pa.string(), nullable=False),
        # Dates : types natifs Parquet
        pa.field("published", pa.timestamp("s", tz="UTC"), nullable=True),
        pa.field("updated", pa.timestamp("s", tz="UTC"), nullable=True),
        pa.field("published_date", pa.date32(), nullable=True),
        pa.field("published_month", pa.string(), nullable=True),
        pa.field("published_year", pa.int32(), nullable=True),
        # Auteurs
        pa.field("authors", pa.list_(pa.string()), nullable=False),
        pa.field("first_author", pa.string(), nullable=True),
        pa.field("authors_count", pa.int32(), nullable=False),
        # Catégories
        pa.field("primary_category", pa.string(), nullable=True),
        pa.field("categories", pa.list_(pa.string()), nullable=False),
        # Liens
        pa.field("pdf_url", pa.string(), nullable=True),
        pa.field("doi", pa.string(), nullable=True),
        pa.field("journal_ref", pa.string(), nullable=True),
        pa.field("comment", pa.string(), nullable=True),
        # Citations Semantic Scholar
        pa.field("citation_count", pa.int32(), nullable=True),
        pa.field("reference_count", pa.int32(), nullable=True),
        pa.field("semantic_scholar_url", pa.string(), nullable=True),
        # Traçabilité collecte (préservée depuis RAW)
        pa.field("_collected_at", pa.string(), nullable=False),
        # Traçabilité traitement
        pa.field("_processed_at", pa.string(), nullable=False),
        # Métadonnées de partition embarquées dans le fichier
        pa.field("_partition_category", pa.string(), nullable=False),
        pa.field("_partition_year", pa.int32(), nullable=False),
        pa.field("_partition_month", pa.int8(), nullable=False),
    ]
)
"""
Schéma Parquet de la zone CLEAN.
"""

# Colonnes du schéma dans l'ordre attendu
_SCHEMA_COLUMNS: list[str] = [field.name for field in CLEAN_SCHEMA]


def prepare_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les conversions de types nécessaires avant écriture Parquet CLEAN.
    """
    df = df.copy()

    # Timestamps UTC : ISO 8601 string (avec TZ) TO datetime64[ns, UTC]
    for col in ("published", "updated"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Date ISO : YYYY-MM-DD string → datetime.date (PyArrow → date32) — vectorisé
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
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def write_clean_file(
    df: pd.DataFrame,
    clean_path: str,
    category: str,
    year: int,
    month: int,
) -> Path:
    """
    Écrit un DataFrame nettoyé en Parquet dans la zone CLEAN.
    """
    if df.empty:
        raise ValueError("Impossible d'écrire un DataFrame vide en zone CLEAN.")

    # Vérifier que toutes les colonnes du schéma sont présentes
    missing = set(_SCHEMA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes pour le schéma CLEAN : {sorted(missing)}. "
        )

    out_path = (
        Path(clean_path)
        / category
        / f"{year:04d}"
        / f"{month:02d}"
        / "articles.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Réordonner les colonnes selon le schéma
    df_ordered = df[_SCHEMA_COLUMNS].copy()

    # Préparer les types (dates natives, downcasts entiers)
    df_ordered = prepare_clean_dataframe(df_ordered)

    table = pa.Table.from_pandas(
        df_ordered,
        schema=CLEAN_SCHEMA,
        preserve_index=False,
        safe=False,  # Autorise les downcasts implicites (int64 → int32/int8, ns → us)
    )
    tmp_path = out_path.with_suffix(".tmp.parquet")
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return out_path
