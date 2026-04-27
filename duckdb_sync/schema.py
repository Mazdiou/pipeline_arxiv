"""
Schéma du data mart dashboard.duckdb : 5 tables.

dim_categories : liste des catégories (ex. cs.AI) avec métadonnées de synchronisation
dim_topics : liste des topics par catégorie, avec métadonnées temporelles
fact_articles : liste des articles par catégorie et topic, avec métadonnées de contenu
fact_topic_monthly : métriques mensuelles par topic (ex. nombre d'articles, nouveauté moyenne)
fact_trend_reports : rapports de tendance mensuels par topic, avec scores composites
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)


_DDL = """
CREATE TABLE IF NOT EXISTS dim_categories (
    category        VARCHAR PRIMARY KEY,
    display_name    VARCHAR        NOT NULL,
    bootstrapped_at TIMESTAMPTZ    NOT NULL,
    last_synced_at  TIMESTAMPTZ    NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_topics (
    topic_id        INTEGER NOT NULL,
    category        VARCHAR NOT NULL,
    topic_label     VARCHAR NOT NULL,
    first_month     VARCHAR NOT NULL,
    last_month      VARCHAR NOT NULL,
    n_months_active INTEGER NOT NULL,
    PRIMARY KEY (topic_id, category)
);

CREATE TABLE IF NOT EXISTS fact_articles (
    arxiv_id              VARCHAR  NOT NULL,
    title                 VARCHAR,
    abstract              VARCHAR,
    first_author          VARCHAR,
    authors_count         INTEGER,
    published_date        DATE,
    updated               DATE,
    published_month       VARCHAR,
    pdf_url               VARCHAR,
    doi                   VARCHAR,
    journal_ref           VARCHAR,
    citation_count        INTEGER,
    reference_count       INTEGER,
    semantic_scholar_url  VARCHAR,
    category              VARCHAR  NOT NULL,
    primary_category      VARCHAR,
    topic_id              INTEGER,
    topic_label           VARCHAR,
    topic_prob            FLOAT,
    is_topic_outlier      BOOLEAN,
    lang                  VARCHAR,
    lang_confidence       FLOAT,
    novelty_score         FLOAT,
    intra_similarity      FLOAT,
    centroid_distance     FLOAT,
    latex_density         FLOAT,
    flesch_kincaid_grade  FLOAT,
    is_short_abstract     BOOLEAN,
    keywords              JSON,
    entity_list           JSON,
    PRIMARY KEY (arxiv_id, category)
);

CREATE TABLE IF NOT EXISTS fact_topic_monthly (
    category               VARCHAR  NOT NULL,
    topic_id               INTEGER  NOT NULL,
    month                  VARCHAR  NOT NULL,
    n_articles             INTEGER  NOT NULL,
    pct_articles           FLOAT,
    mean_novelty           FLOAT,
    mean_topic_prob        FLOAT,
    pct_english            FLOAT,
    mean_centroid_distance FLOAT,
    dominant_lang          VARCHAR,
    mean_flesch_grade      FLOAT,
    mean_latex_density     FLOAT,
    pct_cross_category     FLOAT,
    top_keywords           JSON,
    PRIMARY KEY (category, topic_id, month)
);

CREATE TABLE IF NOT EXISTS fact_trend_reports (
    category                        VARCHAR  NOT NULL,
    topic_id                        INTEGER  NOT NULL,
    month                           VARCHAR  NOT NULL,
    status                          VARCHAR,
    composite_score                 FLOAT,
    trend_score                     FLOAT,
    weak_signal_score               FLOAT,
    semantic_score                  FLOAT,
    z_score_last                    FLOAT,
    ols_slope                       FLOAT,
    mann_kendall_tau                FLOAT,
    mann_kendall_pvalue             FLOAT,
    hurst_exponent                  FLOAT,
    cagr                            FLOAT,
    burst_state_last                INTEGER,
    emergence_score_last            FLOAT,
    novelty_weighted_emergence_last FLOAT,
    persistence_score               FLOAT,
    semantic_drift_last             FLOAT,
    keyword_drift_last              FLOAT,
    stability_score                 FLOAT,
    mean_novelty                    FLOAT,
    mean_topic_prob                 FLOAT,
    mean_centroid_distance          FLOAT,
    pct_english                     FLOAT,
    mean_flesch_grade               FLOAT,
    mean_latex_density              FLOAT,
    pct_cross_category              FLOAT,
    PRIMARY KEY (category, topic_id, month)
);
"""

# Migrations pour les bases existantes : chaque instruction est idempotente.
_MIGRATIONS: list[str] = [
    "ALTER TABLE fact_articles ADD COLUMN IF NOT EXISTS updated               DATE",
    "ALTER TABLE fact_articles ADD COLUMN IF NOT EXISTS primary_category      VARCHAR",
    "ALTER TABLE fact_articles ADD COLUMN IF NOT EXISTS is_topic_outlier      BOOLEAN",
    "ALTER TABLE fact_articles ADD COLUMN IF NOT EXISTS centroid_distance     FLOAT",
    "ALTER TABLE fact_articles ADD COLUMN IF NOT EXISTS flesch_kincaid_grade  FLOAT",
    "ALTER TABLE fact_articles ADD COLUMN IF NOT EXISTS is_short_abstract     BOOLEAN",
    "ALTER TABLE fact_articles DROP COLUMN IF EXISTS word_count",
    "ALTER TABLE fact_topic_monthly ADD COLUMN IF NOT EXISTS mean_topic_prob        FLOAT",
    "ALTER TABLE fact_topic_monthly ADD COLUMN IF NOT EXISTS pct_english            FLOAT",
    "ALTER TABLE fact_topic_monthly ADD COLUMN IF NOT EXISTS mean_centroid_distance  FLOAT",
    "ALTER TABLE fact_topic_monthly ADD COLUMN IF NOT EXISTS dominant_lang          VARCHAR",
    "ALTER TABLE fact_topic_monthly ADD COLUMN IF NOT EXISTS mean_flesch_grade      FLOAT",
    "ALTER TABLE fact_topic_monthly ADD COLUMN IF NOT EXISTS mean_latex_density     FLOAT",
    "ALTER TABLE fact_topic_monthly ADD COLUMN IF NOT EXISTS pct_cross_category     FLOAT",
    "ALTER TABLE fact_trend_reports ADD COLUMN IF NOT EXISTS mean_novelty            FLOAT",
    "ALTER TABLE fact_trend_reports ADD COLUMN IF NOT EXISTS mean_topic_prob         FLOAT",
    "ALTER TABLE fact_trend_reports ADD COLUMN IF NOT EXISTS mean_centroid_distance   FLOAT",
    "ALTER TABLE fact_trend_reports ADD COLUMN IF NOT EXISTS pct_english             FLOAT",
    "ALTER TABLE fact_trend_reports ADD COLUMN IF NOT EXISTS mean_flesch_grade       FLOAT",
    "ALTER TABLE fact_trend_reports ADD COLUMN IF NOT EXISTS mean_latex_density      FLOAT",
    "ALTER TABLE fact_trend_reports ADD COLUMN IF NOT EXISTS pct_cross_category      FLOAT",
]


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Crée toutes les tables si elles n'existent pas encore, puis applique
    les migrations de colonnes sur les bases existantes.

    Idempotent — sans effet si le schéma est déjà complet.
    """
    conn.execute(_DDL)
    _apply_migrations(conn)


def _apply_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Ajoute ou supprime les colonnes nécessaires sur une base existante.

    Chaque ALTER TABLE est exécuté individuellement pour que l'échec
    d'une migration n'empêche pas les suivantes.
    """
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except Exception as exc:
            logger.debug("Migration ignorée (%s…) : %s", stmt[:50], exc)


def delete_category(conn: duckdb.DuckDBPyConnection, category: str) -> None:
    """
    Supprime toutes les données d'une catégorie dans les tables mutables.

    Appelé avant chaque re-sync pour garantir l'idempotence.
    dim_categories est gérée séparément (upsert).
    """
    for table in (
        "dim_topics",
        "fact_articles",
        "fact_topic_monthly",
        "fact_trend_reports",
    ):
        conn.execute(f"DELETE FROM {table} WHERE category = ?", [category])
