"""
Chargement de dim_topics + fact_topic_monthly depuis history.parquet.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import duckdb
import pandas as pd

from ..reader import BaseDataReader

logger = logging.getLogger(__name__)

# Pattern de label générique BERTopic : jamais affiché en production
_GENERIC_LABEL_RE = re.compile(r"^topic_\d+$", re.IGNORECASE)

# Colonnes fact_topic_monthly extraites de history
_MONTHLY_COLUMNS = [
    "category",
    "topic_id",
    "month",
    "n_articles",
    "pct_articles",
    "mean_novelty",
    "mean_topic_prob",
    "pct_english",
    "mean_centroid_distance",
    "dominant_lang",
    "mean_flesch_grade",
    "mean_latex_density",
    "pct_cross_category",
    "top_keywords",
]


def load_topics(
    conn: duckdb.DuckDBPyConnection,
    category: str,
    reader: BaseDataReader,
    topic_labels: Optional[dict[int, str]] = None,
) -> int:
    """
    Lit history.parquet et insère dans dim_topics + fact_topic_monthly.

    Suppose que delete_category() a déjà été appelé pour cette catégorie.

    Args:
        conn         : Connexion DuckDB ouverte.
        category     : Catégorie arXiv (ex: 'stat_ML').
        reader       : Reader abstrait (LocalReader ou MinioReader).
        topic_labels : dict[topic_id → label] issu des rapports de tendance.
                       Les labels descriptifs issus de load_trends() sont
                       prioritaires sur les labels génériques de history.

    Returns:
        Nombre de topics uniques chargés.
    """
    history_df = reader.read_history(category)
    if history_df.empty or "topic_id" not in history_df.columns:
        logger.warning("history.parquet vide ou sans topic_id pour '%s'.", category)
        return 0

    history_df["topic_id"] = pd.to_numeric(
        history_df["topic_id"], errors="coerce"
    ).astype("Int32")
    history_df = history_df.dropna(subset=["topic_id"])

    _load_dim_topics(conn, history_df, category, topic_labels or {})
    _load_fact_topic_monthly(conn, history_df, category)

    n_topics = history_df["topic_id"].nunique()
    logger.info(
        "dim_topics + fact_topic_monthly '%s' : %d topics, %d entrées mensuelles.",
        category,
        n_topics,
        len(history_df),
    )
    return n_topics


def _load_dim_topics(
    conn: duckdb.DuckDBPyConnection,
    history_df: pd.DataFrame,
    category: str,
    topic_labels: dict[int, str],
) -> None:
    """
    Construit et insère dim_topics depuis l'historique.

    Règle de label :
        1. Si topic_labels contient un label descriptif → on l'utilise.
        2. Sinon, on cherche dans history (colonne topic_label si présente).
        3. En dernier recours : fallback "Topic {id}" (lisible, pas générique).

    L'INSERT utilise ON CONFLICT DO UPDATE avec une condition sur le label
    entrant : un label générique n'écrase jamais un label descriptif existant.
    Cela protège les labels acquis lors d'un run précédent.
    """
    rows = []
    for topic_id, group in history_df.groupby("topic_id"):
        tid = int(topic_id)
        months = sorted(group["month"].dropna().unique().tolist())

        # Résolution du label (priorité décroissante)
        label = _resolve_label(tid, topic_labels, group)

        rows.append(
            {
                "topic_id": tid,
                "category": category,
                "topic_label": label,
                "first_month": months[0] if months else "",
                "last_month": months[-1] if months else "",
                "n_months_active": len(months),
            }
        )

    if not rows:
        return

    dim_df = pd.DataFrame(rows)
    conn.register("_dim_topics_batch", dim_df)

    conn.execute("""
        INSERT INTO dim_topics
            (topic_id, category, topic_label, first_month, last_month, n_months_active)
        SELECT topic_id, category, topic_label, first_month, last_month, n_months_active
        FROM _dim_topics_batch
        ON CONFLICT (topic_id, category) DO UPDATE SET
            topic_label     = CASE
                WHEN excluded.topic_label NOT LIKE 'topic_%'
                     OR dim_topics.topic_label LIKE 'topic_%'
                THEN excluded.topic_label
                ELSE dim_topics.topic_label
            END,
            last_month      = excluded.last_month,
            n_months_active = excluded.n_months_active
    """)
    conn.unregister("_dim_topics_batch")


def _load_fact_topic_monthly(
    conn: duckdb.DuckDBPyConnection,
    history_df: pd.DataFrame,
    category: str,
) -> None:
    """Construit et insère fact_topic_monthly depuis l'historique."""
    monthly_df = history_df.copy()
    monthly_df["category"] = category

    if "top_keywords" in monthly_df.columns:
        monthly_df["top_keywords"] = monthly_df["top_keywords"].apply(
            _serialize_keywords_to_json
        )

    available = [c for c in _MONTHLY_COLUMNS if c in monthly_df.columns]
    monthly_df = monthly_df[available]

    col_list = ", ".join(monthly_df.columns.tolist())
    conn.register("_monthly_batch", monthly_df)
    conn.execute(
        f"INSERT INTO fact_topic_monthly ({col_list}) SELECT * FROM _monthly_batch"
    )
    conn.unregister("_monthly_batch")


def _resolve_label(
    topic_id: int,
    topic_labels: dict[int, str],
    group: pd.DataFrame,
) -> str:
    """
    Détermine le meilleur label disponible pour un topic_id.

    Priorité :
        1. Label descriptif depuis topic_labels (issu des rapports de tendance)
        2. Label depuis la colonne topic_label de history (si disponible et descriptif)
        3. Fallback lisible : "Topic {id}"

    Args:
        topic_id     : Identifiant du topic.
        topic_labels : Labels extraits des rapports fact_trend_reports.
        group        : Sous-DataFrame history du topic.

    Returns:
        Label résolu, jamais générique (topic_N).
    """
    # 1. Label issu des rapports (le plus fiable)
    label_from_report = topic_labels.get(topic_id)
    if label_from_report and not _GENERIC_LABEL_RE.match(str(label_from_report)):
        return label_from_report

    # 2. Label depuis history (si la colonne existe)
    if "topic_label" in group.columns:
        history_labels = group["topic_label"].dropna().unique().tolist()
        # Prendre le dernier non-générique
        for candidate in reversed(history_labels):
            if candidate and not _GENERIC_LABEL_RE.match(str(candidate)):
                return str(candidate)

    # 3. Fallback lisible : pas "topic_2" mais "Topic 2"
    return f"Topic {topic_id}"


def _serialize_keywords_to_json(value) -> str | None:
    """Sérialise la liste de mots-clés en JSON string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value), ensure_ascii=False)
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
