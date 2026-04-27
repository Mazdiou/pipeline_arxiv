"""Chargement des rapports de tendance dans fact_trend_reports."""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

from ..reader import BaseDataReader

logger = logging.getLogger(__name__)

# Colonnes fact_trend_reports (correspondance avec schema.py)
_REPORT_COLUMNS = [
    "category",
    "topic_id",
    "month",
    "status",
    "composite_score",
    "trend_score",
    "weak_signal_score",
    "semantic_score",
    "z_score_last",
    "ols_slope",
    "mann_kendall_tau",
    "mann_kendall_pvalue",
    "hurst_exponent",
    "cagr",
    "burst_state_last",
    "emergence_score_last",
    "novelty_weighted_emergence_last",
    "persistence_score",
    "semantic_drift_last",
    "keyword_drift_last",
    "stability_score",
    "mean_novelty",
    "mean_topic_prob",
    "mean_centroid_distance",
    "pct_english",
    "mean_flesch_grade",
    "mean_latex_density",
    "pct_cross_category",
]


def load_trends(
    conn: duckdb.DuckDBPyConnection,
    category: str,
    reader: BaseDataReader,
) -> dict[int, str]:
    """
    Lit tous les rapports Parquet d'une catégorie et insère dans fact_trend_reports.

    Suppose que delete_category() a déjà été appelé pour cette catégorie.

    Args:
        conn     : Connexion DuckDB ouverte.
        category : Catégorie arXiv (ex: 'stat_ML').
        reader   : Reader abstrait (LocalReader ou MinioReader).

    Returns:
        dict[topic_id → topic_label] avec le label le plus récent par topic.
        Utilisé par topics.py pour peupler dim_topics avec les labels descriptifs.
    """
    raw_frames = reader.read_reports(category)
    if not raw_frames:
        logger.warning("Aucun rapport Parquet trouvé pour '%s'.", category)
        return {}

    all_frames: list[pd.DataFrame] = []
    for df in raw_frames:
        normalized = _normalize_report(df, category)
        if not normalized.empty:
            all_frames.append(normalized)

    if not all_frames:
        return {}

    combined_df = pd.concat(all_frames, ignore_index=True)

    # Déduplication sur la clé primaire (category, topic_id, month)
    combined_df = combined_df.drop_duplicates(
        subset=["category", "topic_id", "month"], keep="last"
    )

    # Extraire les labels descriptifs AVANT insertion
    topic_labels = _extract_latest_labels(combined_df)

    available = [c for c in _REPORT_COLUMNS if c in combined_df.columns]
    insert_df = combined_df[available]

    col_list = ", ".join(insert_df.columns.tolist())
    conn.register("_trends_batch", insert_df)
    conn.execute(
        f"INSERT INTO fact_trend_reports ({col_list}) SELECT * FROM _trends_batch"
    )
    conn.unregister("_trends_batch")

    logger.info(
        "fact_trend_reports '%s' : %d entrées insérées depuis %d rapports.",
        category,
        len(insert_df),
        len(raw_frames),
    )
    return topic_labels


def _normalize_report(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Normalise un DataFrame rapport brut (colonnes, types).

    Args:
        df       : DataFrame rapport brut lu depuis le reader.
        category : Catégorie arXiv (ajoutée si absente).

    Returns:
        DataFrame normalisé ou vide.
    """
    if df.empty:
        return pd.DataFrame()

    # 'period' : 'month' (convention data mart)
    if "period" in df.columns and "month" not in df.columns:
        df = df.rename(columns={"period": "month"})

    if "category" not in df.columns:
        df = df.copy()
        df["category"] = category

    if "topic_id" in df.columns:
        df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").astype("Int32")

    return df


def _extract_latest_labels(df: pd.DataFrame) -> dict[int, str]:
    """
    Extrait le label le plus récent par topic_id depuis les rapports combinés.

    'Le plus récent' = rapport avec le mois (month) le plus élevé en ordre lexicographique
    (format YYYY-MM, tri alphabétique = tri chronologique).

    Args:
        df : DataFrame combiné de tous les rapports.

    Returns:
        dict[topic_id → topic_label].
    """
    if "topic_label" not in df.columns or "topic_id" not in df.columns:
        return {}

    latest = (
        df.sort_values("month", ascending=True)
        .groupby("topic_id")["topic_label"]
        .last()
    )
    return {int(tid): str(label) for tid, label in latest.items()}
