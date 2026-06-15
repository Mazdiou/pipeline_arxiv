from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)


def _duckdb_path() -> str:
    return os.getenv("DUCKDB_PATH", "data/dashboard.duckdb")


def _close(conn) -> None:
    """Ferme silencieusement une connexion DuckDB (tolère None et les erreurs)."""
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def db_exists() -> bool:
    """Vérifie que le fichier dashboard.duckdb existe et est non-vide."""
    p = Path(_duckdb_path())
    return p.exists() and p.stat().st_size > 0


def _connect(read_only: bool = True, retries: int = 3) -> duckdb.DuckDBPyConnection:
    """
    Ouvre une connexion DuckDB avec retry (utile sur Azure VM avec I/O lents).

    Raises:
        FileNotFoundError : si dashboard.duckdb est absent.
        duckdb.IOException : si le fichier est corrompu après tous les retries.
    """
    path = _duckdb_path()
    if not Path(path).exists():
        raise FileNotFoundError(
            f"dashboard.duckdb introuvable : {path}. "
            "Lancez une première sync pour initialiser la base."
        )
    last_exc: Exception = RuntimeError("connect failed")
    for attempt in range(retries):
        try:
            return duckdb.connect(path, read_only=read_only)
        except duckdb.IOException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise last_exc


@dataclass
class CategoryRow:
    """
    Ligne dim_categories enrichie avec les statistiques agrégées.
    Utilisée pour afficher les cards sur la Page 0.
    """

    category: str
    display_name: str
    bootstrapped_at: Optional[datetime]
    last_synced_at: Optional[datetime]
    n_articles: int
    n_topics: int
    n_months: int

    @property
    def status(self) -> str:
        if self.last_synced_at is None:
            return "error"
        now = datetime.now(tz=timezone.utc)
        try:
            synced = self.last_synced_at
            if synced.tzinfo is None:
                synced = synced.replace(tzinfo=timezone.utc)
            delta = now - synced
            if delta < timedelta(hours=24):
                return "ok"
            if delta < timedelta(days=7):
                return "warning"
            return "error"
        except Exception:
            return "error"

    @property
    def status_label(self) -> str:
        return {"ok": "À jour", "warning": "Sync en attente", "error": "Désynchronisé"}[
            self.status
        ]

    @property
    def status_color(self) -> str:
        return {"ok": "green", "warning": "orange", "error": "red"}[self.status]

    @property
    def last_synced_label(self) -> str:
        if self.last_synced_at is None:
            return "Jamais synchronisée"
        try:
            synced = self.last_synced_at
            if synced.tzinfo is None:
                synced = synced.replace(tzinfo=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            delta = now - synced
            total_seconds = int(delta.total_seconds())
            if total_seconds < 60:
                return "à l'instant"
            if total_seconds < 3600:
                return f"il y a {total_seconds // 60} min"
            if delta < timedelta(hours=24):
                return f"il y a {total_seconds // 3600}h"
            return f"il y a {delta.days}j"
        except Exception:
            return str(self.last_synced_at)[:16]


@dataclass
class CategoryCounts:
    """Compteurs détaillés par catégorie — utilisés dans la modale de suppression."""

    n_articles: int = 0
    n_topics: int = 0
    n_trend_reports: int = 0
    n_monthly_entries: int = 0


# --------------------------------------------------------------------------- #
# Requêtes globales                                                            #
# --------------------------------------------------------------------------- #


def get_global_stats() -> dict:
    if not db_exists():
        return {"n_categories": 0, "n_articles": 0, "last_sync": None}
    conn = None
    try:
        conn = _connect()
        n_cats = conn.execute("SELECT COUNT(*) FROM dim_categories").fetchone()[0]
        n_arts = conn.execute("SELECT COUNT(*) FROM fact_articles").fetchone()[0]
        last_sync = conn.execute(
            "SELECT MAX(last_synced_at) FROM dim_categories"
        ).fetchone()[0]
        return {"n_categories": n_cats, "n_articles": n_arts, "last_sync": last_sync}
    except Exception as exc:
        logger.error("get_global_stats échoué : %s", exc)
        return {"n_categories": 0, "n_articles": 0, "last_sync": None}
    finally:
        _close(conn)


def get_all_category_rows() -> list[CategoryRow]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """
            SELECT
                c.category,
                c.display_name,
                c.bootstrapped_at,
                c.last_synced_at,
                COALESCE(a.n_articles, 0)  AS n_articles,
                COALESCE(t.n_topics,   0)  AS n_topics,
                COALESCE(m.n_months,   0)  AS n_months
            FROM dim_categories c
            LEFT JOIN (
                SELECT category, COUNT(*) AS n_articles
                FROM fact_articles
                GROUP BY category
            ) a USING (category)
            LEFT JOIN (
                SELECT category, COUNT(DISTINCT topic_id) AS n_topics
                FROM dim_topics
                GROUP BY category
            ) t USING (category)
            LEFT JOIN (
                SELECT category, COUNT(DISTINCT month) AS n_months
                FROM fact_topic_monthly
                GROUP BY category
            ) m USING (category)
            ORDER BY c.category
        """
        ).fetchall()
        return [
            CategoryRow(
                category=r[0],
                display_name=r[1],
                bootstrapped_at=r[2],
                last_synced_at=r[3],
                n_articles=int(r[4]),
                n_topics=int(r[5]),
                n_months=int(r[6]),
            )
            for r in rows
        ]
    except Exception as exc:
        logger.error("get_all_category_rows échoué : %s", exc)
        return []
    finally:
        _close(conn)


# --------------------------------------------------------------------------- #
# Requêtes par catégorie                                                       #
# --------------------------------------------------------------------------- #


def get_category_counts(category: str) -> CategoryCounts:
    if not db_exists():
        return CategoryCounts()
    conn = None
    try:
        conn = _connect()
        n_art = conn.execute(
            "SELECT COUNT(*) FROM fact_articles WHERE category = ?", [category]
        ).fetchone()[0]
        n_top = conn.execute(
            "SELECT COUNT(DISTINCT topic_id) FROM dim_topics WHERE category = ?",
            [category],
        ).fetchone()[0]
        n_tr = conn.execute(
            "SELECT COUNT(*) FROM fact_trend_reports WHERE category = ?", [category]
        ).fetchone()[0]
        n_mo = conn.execute(
            "SELECT COUNT(*) FROM fact_topic_monthly WHERE category = ?", [category]
        ).fetchone()[0]
        return CategoryCounts(
            n_articles=int(n_art),
            n_topics=int(n_top),
            n_trend_reports=int(n_tr),
            n_monthly_entries=int(n_mo),
        )
    except Exception as exc:
        logger.error("get_category_counts '%s' échoué : %s", category, exc)
        return CategoryCounts()
    finally:
        _close(conn)


# --------------------------------------------------------------------------- #
# Page 1 — Vue catégorie                                                       #
# --------------------------------------------------------------------------- #


def get_categories_with_data() -> list[tuple[str, str]]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT category, display_name FROM dim_categories ORDER BY display_name"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception as exc:
        logger.error("get_categories_with_data échoué : %s", exc)
        return []
    finally:
        _close(conn)


def get_category_overview(category: str) -> dict:
    if not db_exists():
        return {}
    conn = None
    try:
        conn = _connect()
        n_art = conn.execute(
            "SELECT COUNT(*) FROM fact_articles WHERE category = ?", [category]
        ).fetchone()[0]
        n_top = conn.execute(
            "SELECT COUNT(DISTINCT topic_id) FROM dim_topics WHERE category = ?",
            [category],
        ).fetchone()[0]
        n_mon = conn.execute(
            "SELECT COUNT(DISTINCT month) FROM fact_topic_monthly WHERE category = ?",
            [category],
        ).fetchone()[0]
        avg_nov = conn.execute(
            "SELECT AVG(novelty_score) FROM fact_articles WHERE category = ? AND novelty_score IS NOT NULL",
            [category],
        ).fetchone()[0]
        dates = conn.execute(
            "SELECT MIN(published_month), MAX(published_month) FROM fact_articles WHERE category = ?",
            [category],
        ).fetchone()
        pct_en = conn.execute(
            """SELECT COUNT(*) FILTER (WHERE lang = 'en') * 100.0 / NULLIF(COUNT(*), 0)
               FROM fact_articles WHERE category = ?""",
            [category],
        ).fetchone()[0]
        return {
            "n_articles": int(n_art),
            "n_topics": int(n_top),
            "n_months": int(n_mon),
            "avg_novelty": round(float(avg_nov), 3) if avg_nov is not None else None,
            "date_min": dates[0],
            "date_max": dates[1],
            "pct_english": round(float(pct_en), 1) if pct_en is not None else None,
        }
    except Exception as exc:
        logger.error("get_category_overview '%s' échoué : %s", category, exc)
        return {}
    finally:
        _close(conn)


def get_monthly_volumes(category: str) -> list[dict]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT published_month, COUNT(*) AS n
               FROM fact_articles
               WHERE category = ? AND published_month IS NOT NULL
               GROUP BY published_month
               ORDER BY published_month""",
            [category],
        ).fetchall()
        return [{"month": r[0], "n_articles": int(r[1])} for r in rows]
    except Exception as exc:
        logger.error("get_monthly_volumes '%s' échoué : %s", category, exc)
        return []
    finally:
        _close(conn)


def get_top_topics(category: str, limit: int = 20) -> list[dict]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()

        rows = conn.execute(
            """
            SELECT
                t.topic_id,
                
                COALESCE(
                    ANY_VALUE(a2.topic_label),
                    t.topic_label,
                    'Topic ' || t.topic_id
                ) AS topic_label,

                COALESCE(a.n_articles, 0) AS n_articles,
                t.n_months_active,
                tr.status AS last_status

            FROM dim_topics t

            LEFT JOIN (
                SELECT topic_id, COUNT(*) AS n_articles
                FROM fact_articles
                WHERE category = ?
                GROUP BY topic_id
            ) a ON t.topic_id = a.topic_id

            LEFT JOIN fact_articles a2
                ON t.topic_id = a2.topic_id
               AND t.category = a2.category

            LEFT JOIN (
                SELECT topic_id, status
                FROM fact_trend_reports
                WHERE category = ?
                  AND month = (
                      SELECT MAX(month)
                      FROM fact_trend_reports
                      WHERE category = ?
                  )
            ) tr ON t.topic_id = tr.topic_id

            WHERE t.category = ?
              AND t.topic_id >= 0

            GROUP BY
                t.topic_id,
                t.n_months_active,
                tr.status,
                a.n_articles,
                t.topic_label

            ORDER BY n_articles DESC
            LIMIT ?
            """,
            [category, category, category, category, limit],
        ).fetchall()

        return [
            {
                "topic_id": int(r[0]),
                "topic_label": r[1] or f"Topic {r[0]}",
                "n_articles": int(r[2]),
                "n_months_active": int(r[3]),
                "last_status": r[4],
            }
            for r in rows
        ]

    except Exception as exc:
        logger.error("get_top_topics '%s' échoué : %s", category, exc)
        return []
    finally:
        _close(conn)


def get_lang_distribution(category: str) -> list[dict]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT COALESCE(lang, 'inconnu') AS lang, COUNT(*) AS n
               FROM fact_articles
               WHERE category = ?
               GROUP BY lang
               ORDER BY n DESC""",
            [category],
        ).fetchall()
        return [{"lang": r[0], "n_articles": int(r[1])} for r in rows]
    except Exception as exc:
        logger.error("get_lang_distribution '%s' échoué : %s", category, exc)
        return []
    finally:
        _close(conn)


def get_status_counts(category: str) -> dict[str, int]:
    if not db_exists():
        return {}
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT status, COUNT(*) AS n
               FROM fact_trend_reports
               WHERE category = ?
                 AND month = (
                     SELECT MAX(month) FROM fact_trend_reports WHERE category = ?
                 )
               GROUP BY status
               ORDER BY n DESC""",
            [category, category],
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows if r[0] is not None}
    except Exception as exc:
        logger.error("get_status_counts '%s' échoué : %s", category, exc)
        return {}
    finally:
        _close(conn)


# --------------------------------------------------------------------------- #
# Page 4 — Articles                                                            #
# --------------------------------------------------------------------------- #


def get_articles_page(
    category: str,
    *,
    topic_id: Optional[int] = None,
    lang: Optional[str] = None,
    month_from: Optional[str] = None,
    month_to: Optional[str] = None,
    novelty_min: Optional[float] = None,
    search: Optional[str] = None,
    exclude_outliers: bool = False,
    exclude_short: bool = False,
    limit: int = 25,
    offset: int = 0,
) -> list[dict]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        where, params = _articles_where(
            category,
            topic_id,
            lang,
            month_from,
            month_to,
            novelty_min,
            search,
            exclude_outliers,
            exclude_short,
        )
        rows = conn.execute(
            f"""SELECT arxiv_id, title, first_author, authors_count,
                       published_date, updated, primary_category,
                       topic_label, lang,
                       novelty_score, intra_similarity, centroid_distance,
                       is_topic_outlier,
                       citation_count, topic_prob, latex_density,
                       flesch_kincaid_grade, is_short_abstract,
                       pdf_url, keywords, entity_list
                FROM fact_articles
                WHERE {where}
                ORDER BY published_month DESC NULLS LAST, arxiv_id
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
        return [
            {
                "arxiv_id": r[0],
                "title": r[1] or "",
                "first_author": r[2] or "",
                "authors_count": int(r[3]) if r[3] is not None else None,
                "published_date": str(r[4]) if r[4] else "",
                "updated": str(r[5]) if r[5] else "",
                "primary_category": r[6] or "",
                "topic_label": r[7] or "",
                "lang": r[8] or "",
                "novelty_score": round(float(r[9]), 4) if r[9] is not None else None,
                "intra_similarity": (
                    round(float(r[10]), 4) if r[10] is not None else None
                ),
                "centroid_distance": (
                    round(float(r[11]), 4) if r[11] is not None else None
                ),
                "is_topic_outlier": bool(r[12]) if r[12] is not None else False,
                "citation_count": int(r[13]) if r[13] is not None else None,
                "topic_prob": round(float(r[14]), 4) if r[14] is not None else None,
                "latex_density": round(float(r[15]), 4) if r[15] is not None else None,
                "flesch_kincaid_grade": (
                    round(float(r[16]), 2) if r[16] is not None else None
                ),
                "is_short_abstract": bool(r[17]) if r[17] is not None else False,
                "pdf_url": r[18] or "",
                "keywords": r[19],
                "entity_list": r[20],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("get_articles_page '%s' échoué : %s", category, exc)
        return []
    finally:
        _close(conn)


def get_articles_count(
    category: str,
    *,
    topic_id: Optional[int] = None,
    lang: Optional[str] = None,
    month_from: Optional[str] = None,
    month_to: Optional[str] = None,
    novelty_min: Optional[float] = None,
    search: Optional[str] = None,
    exclude_outliers: bool = False,
    exclude_short: bool = False,
) -> int:
    if not db_exists():
        return 0
    conn = None
    try:
        conn = _connect()
        where, params = _articles_where(
            category,
            topic_id,
            lang,
            month_from,
            month_to,
            novelty_min,
            search,
            exclude_outliers,
            exclude_short,
        )
        n = conn.execute(
            f"SELECT COUNT(*) FROM fact_articles WHERE {where}", params
        ).fetchone()[0]
        return int(n)
    except Exception as exc:
        logger.error("get_articles_count '%s' échoué : %s", category, exc)
        return 0
    finally:
        _close(conn)


def get_article_detail(arxiv_id: str, category: str) -> dict:
    """
    Mapping complet :
        row[0]  arxiv_id            row[15] topic_label
        row[1]  title               row[16] topic_prob
        row[2]  abstract            row[17] is_topic_outlier
        row[3]  first_author        row[18] primary_category
        row[4]  authors_count       row[19] lang
        row[5]  published_date      row[20] lang_confidence
        row[6]  updated             row[21] novelty_score
        row[7]  published_month     row[22] intra_similarity
        row[8]  pdf_url             row[23] centroid_distance
        row[10] journal_ref         row[25] latex_density
        row[11] citation_count      row[26] flesch_kincaid_grade
        row[12] reference_count     row[27] is_short_abstract
        row[13] semantic_scholar_url row[28] keywords
        row[14] topic_id            row[29] entity_list
    """
    if not db_exists():
        return {}
    conn = None
    try:
        conn = _connect()
        row = conn.execute(
            """SELECT arxiv_id, title, abstract, first_author, authors_count,
                      published_date, updated, published_month, pdf_url,
                      doi, journal_ref, citation_count, reference_count,
                      semantic_scholar_url,
                      topic_id, topic_label, topic_prob, is_topic_outlier,
                      primary_category,
                      lang, lang_confidence,
                      novelty_score, intra_similarity, centroid_distance,
                      latex_density, flesch_kincaid_grade, is_short_abstract,
                      keywords, entity_list
               FROM fact_articles
               WHERE arxiv_id = ? AND category = ?""",
            [arxiv_id, category],
        ).fetchone()
        if not row:
            return {}

        def _f(v):
            return round(float(v), 4) if v is not None else None

        return {
            "arxiv_id": row[0],
            "title": row[1] or "",
            "abstract": row[2] or "",
            "first_author": row[3] or "",
            "authors_count": int(row[4]) if row[4] is not None else None,
            "published_date": str(row[5]) if row[5] else "",
            "updated": str(row[6]) if row[6] else "",
            "published_month": row[7] or "",
            "pdf_url": row[8] or "",
            "doi": row[9] or "",
            "journal_ref": row[10] or "",
            "citation_count": int(row[11]) if row[11] is not None else None,
            "reference_count": int(row[12]) if row[12] is not None else None,
            "semantic_scholar_url": row[13] or "",
            "topic_id": row[14],
            "topic_label": row[15] or "",
            "topic_prob": _f(row[16]),
            "is_topic_outlier": bool(row[17]) if row[17] is not None else False,
            "primary_category": row[18] or "",
            "lang": row[19] or "",
            "lang_confidence": _f(row[20]),
            "novelty_score": _f(row[21]),
            "intra_similarity": _f(row[22]),
            "centroid_distance": _f(row[23]),
            "latex_density": _f(row[24]),
            "flesch_kincaid_grade": (
                round(float(row[25]), 2) if row[25] is not None else None
            ),
            "is_short_abstract": bool(row[26]) if row[26] is not None else False,
            "keywords": row[27],
            "entity_list": row[28],
        }
    except Exception as exc:
        logger.error("get_article_detail '%s' échoué : %s", arxiv_id, exc)
        return {}
    finally:
        _close(conn)


def get_distinct_langs(category: str) -> list[str]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT lang, COUNT(*) AS n
               FROM fact_articles
               WHERE category = ? AND lang IS NOT NULL
               GROUP BY lang ORDER BY n DESC""",
            [category],
        ).fetchall()
        return [r[0] for r in rows]
    except Exception as exc:
        logger.error("get_distinct_langs '%s' échoué : %s", category, exc)
        return []
    finally:
        _close(conn)


def _articles_where(
    category: str,
    topic_id: Optional[int],
    lang: Optional[str],
    month_from: Optional[str],
    month_to: Optional[str],
    novelty_min: Optional[float],
    search: Optional[str],
    exclude_outliers: bool = False,
    exclude_short: bool = False,
) -> tuple[str, list]:
    clauses = ["category = ?"]
    params: list = [category]

    if topic_id is not None:
        clauses.append("topic_id = ?")
        params.append(topic_id)
    if lang:
        clauses.append("lang = ?")
        params.append(lang)
    if month_from:
        clauses.append("published_month >= ?")
        params.append(month_from)
    if month_to:
        clauses.append("published_month <= ?")
        params.append(month_to)
    if novelty_min is not None and novelty_min > 0:
        clauses.append("novelty_score >= ?")
        params.append(novelty_min)
    if search:
        clauses.append("title ILIKE ?")
        params.append(f"%{search}%")
    if exclude_outliers:
        clauses.append("(is_topic_outlier IS NULL OR is_topic_outlier = FALSE)")
    if exclude_short:
        clauses.append("(is_short_abstract IS NULL OR is_short_abstract = FALSE)")

    return " AND ".join(clauses), params


# --------------------------------------------------------------------------- #
# Page 3 — Analyse d'un topic                                                  #
# --------------------------------------------------------------------------- #


def get_topics_for_category(category: str) -> list[dict]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT
                   t.topic_id,
                   COALESCE(
                       (SELECT topic_label
                        FROM fact_articles
                        WHERE category = t.category AND topic_id = t.topic_id
                          AND topic_label IS NOT NULL
                        LIMIT 1),
                       t.topic_label,
                       'Topic ' || t.topic_id
                   ) AS topic_label,
                   t.n_months_active,
                   tr.status,
                   tr.composite_score,
                   COALESCE(a.n_articles, 0) AS n_articles
               FROM dim_topics t
               LEFT JOIN fact_trend_reports tr
                   ON t.topic_id = tr.topic_id AND t.category = tr.category
                   AND tr.month = (
                       SELECT MAX(month) FROM fact_trend_reports
                       WHERE category = ? AND topic_id = t.topic_id
                   )
               LEFT JOIN (
                   SELECT topic_id, COUNT(*) AS n_articles
                   FROM fact_articles WHERE category = ?
                   GROUP BY topic_id
               ) a ON t.topic_id = a.topic_id
               WHERE t.category = ? AND t.topic_id >= 0
               ORDER BY COALESCE(tr.composite_score, -999) DESC""",
            [category, category, category],
        ).fetchall()
        return [
            {
                "topic_id": int(r[0]),
                "topic_label": r[1] or f"Topic {r[0]}",
                "n_months_active": int(r[2]),
                "status": r[3],
                "composite_score": round(float(r[4]), 4) if r[4] is not None else None,
                "n_articles": int(r[5]),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("get_topics_for_category '%s' échoué : %s", category, exc)
        return []
    finally:
        _close(conn)


def get_topic_detail(category: str, topic_id: int) -> dict:
    """
    CORRECTIONS :
        - JOIN → LEFT JOIN sur fact_trend_reports
        - MAX(month) filtré sur topic_id (pas global catégorie)
        - 6 paramètres au lieu de 5
        - topic_label amélioré avec fallback depuis fact_articles
    """
    if not db_exists():
        return {}
    conn = None
    try:
        conn = _connect()
        row = conn.execute(
            """SELECT
                   COALESCE(
                       (SELECT topic_label
                        FROM fact_articles
                        WHERE category = t.category AND topic_id = t.topic_id
                          AND topic_label IS NOT NULL
                        LIMIT 1),
                       t.topic_label,
                       'Topic ' || t.topic_id
                   ) AS topic_label,
                   t.first_month,
                   t.last_month,
                   t.n_months_active,
                   tr.month,
                   tr.status,
                   tr.composite_score,
                   tr.trend_score,
                   tr.weak_signal_score,
                   tr.semantic_score,
                   tr.z_score_last,
                   tr.ols_slope,
                   tr.mann_kendall_tau,
                   tr.mann_kendall_pvalue,
                   tr.hurst_exponent,
                   tr.cagr,
                   tr.burst_state_last,
                   tr.emergence_score_last,
                   tr.novelty_weighted_emergence_last,
                   tr.persistence_score,
                   tr.semantic_drift_last,
                   tr.keyword_drift_last,
                   tr.stability_score,
                   tr.mean_novelty,
                   tr.mean_topic_prob,
                   tr.mean_centroid_distance,
                   tr.pct_english,
                   tr.mean_flesch_grade,
                   tr.mean_latex_density,
                   tr.pct_cross_category,
                   COALESCE(a.n_articles, 0) AS n_articles
               FROM dim_topics t
               LEFT JOIN fact_trend_reports tr
                   ON t.topic_id = tr.topic_id AND t.category = tr.category
                   AND tr.month = (
                       SELECT MAX(month) FROM fact_trend_reports
                       WHERE category = ? AND topic_id = ?
                   )
               LEFT JOIN (
                   SELECT COUNT(*) AS n_articles
                   FROM fact_articles WHERE category = ? AND topic_id = ?
               ) a ON TRUE
               WHERE t.topic_id = ? AND t.category = ?""",
            [category, topic_id, category, topic_id, topic_id, category],
        ).fetchone()
        if not row:
            return {}

        def _f(v):
            return round(float(v), 4) if v is not None else None

        return {
            "topic_label": row[0] or f"Topic {topic_id}",
            "first_month": row[1],
            "last_month": row[2],
            "n_months_active": int(row[3]),
            "report_month": row[4],
            "status": row[5],
            "composite_score": _f(row[6]),
            "trend_score": _f(row[7]),
            "weak_signal_score": _f(row[8]),
            "semantic_score": _f(row[9]),
            "z_score_last": _f(row[10]),
            "ols_slope": _f(row[11]),
            "mann_kendall_tau": _f(row[12]),
            "mann_kendall_pvalue": _f(row[13]),
            "hurst_exponent": _f(row[14]),
            "cagr": _f(row[15]),
            "burst_state_last": int(row[16]) if row[16] is not None else None,
            "emergence_score_last": _f(row[17]),
            "novelty_weighted_emergence_last": _f(row[18]),
            "persistence_score": _f(row[19]),
            "semantic_drift_last": _f(row[20]),
            "keyword_drift_last": _f(row[21]),
            "stability_score": _f(row[22]),
            "mean_novelty": _f(row[23]),
            "mean_topic_prob": _f(row[24]),
            "mean_centroid_distance": _f(row[25]),
            "pct_english": round(float(row[26]), 1) if row[26] is not None else None,
            "mean_flesch_grade": (
                round(float(row[27]), 2) if row[27] is not None else None
            ),
            "mean_latex_density": _f(row[28]),
            "pct_cross_category": (
                round(float(row[29]), 1) if row[29] is not None else None
            ),
            "n_articles": int(row[30]),
        }
    except Exception as exc:
        logger.error("get_topic_detail '%s'/%s échoué : %s", category, topic_id, exc)
        return {}
    finally:
        _close(conn)


def get_topic_monthly_stats(category: str, topic_id: int) -> list[dict]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT month, n_articles, pct_articles, mean_novelty,
                      mean_topic_prob, pct_english, mean_centroid_distance,
                      dominant_lang, mean_flesch_grade, mean_latex_density,
                      pct_cross_category, top_keywords
               FROM fact_topic_monthly
               WHERE category = ? AND topic_id = ?
               ORDER BY month""",
            [category, topic_id],
        ).fetchall()
        return [
            {
                "month": r[0],
                "n_articles": int(r[1]),
                "pct_articles": round(float(r[2]), 2) if r[2] is not None else None,
                "mean_novelty": round(float(r[3]), 4) if r[3] is not None else None,
                "mean_topic_prob": round(float(r[4]), 4) if r[4] is not None else None,
                "pct_english": round(float(r[5]), 1) if r[5] is not None else None,
                "mean_centroid_distance": (
                    round(float(r[6]), 4) if r[6] is not None else None
                ),
                "dominant_lang": r[7] or "",
                "mean_flesch_grade": (
                    round(float(r[8]), 2) if r[8] is not None else None
                ),
                "mean_latex_density": (
                    round(float(r[9]), 4) if r[9] is not None else None
                ),
                "pct_cross_category": (
                    round(float(r[10]), 1) if r[10] is not None else None
                ),
                "top_keywords": r[11],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(
            "get_topic_monthly_stats '%s'/%s échoué : %s", category, topic_id, exc
        )
        return []
    finally:
        _close(conn)


def get_trend_full_history(category: str, topic_id: int) -> list[dict]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT month, status, composite_score, trend_score,
                      weak_signal_score, semantic_score,
                      z_score_last, ols_slope, cagr,
                      emergence_score_last, persistence_score,
                      semantic_drift_last, keyword_drift_last, stability_score
               FROM fact_trend_reports
               WHERE category = ? AND topic_id = ?
               ORDER BY month""",
            [category, topic_id],
        ).fetchall()

        def _f(v):
            return round(float(v), 4) if v is not None else None

        return [
            {
                "month": r[0],
                "status": r[1],
                "composite_score": _f(r[2]),
                "trend_score": _f(r[3]),
                "weak_signal_score": _f(r[4]),
                "semantic_score": _f(r[5]),
                "z_score_last": _f(r[6]),
                "ols_slope": _f(r[7]),
                "cagr": _f(r[8]),
                "emergence_score_last": _f(r[9]),
                "persistence_score": _f(r[10]),
                "semantic_drift_last": _f(r[11]),
                "keyword_drift_last": _f(r[12]),
                "stability_score": _f(r[13]),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(
            "get_trend_full_history '%s'/%s échoué : %s", category, topic_id, exc
        )
        return []
    finally:
        _close(conn)


# --------------------------------------------------------------------------- #
# Page 2 — Radar des tendances                                                 #
# --------------------------------------------------------------------------- #


def get_available_trend_months(category: str) -> list[str]:
    """
    AJOUT : retourne les mois disponibles pour le sélecteur de la Page 2.
    """
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT DISTINCT month
               FROM fact_trend_reports
               WHERE category = ?
               ORDER BY month DESC""",
            [category],
        ).fetchall()
        return [r[0] for r in rows]
    except Exception as exc:
        logger.error("get_available_trend_months '%s' échoué : %s", category, exc)
        return []
    finally:
        _close(conn)


def get_trends_overview(category: str, month: Optional[str] = None) -> list[dict]:
    """
    CORRIGÉ :
        - label topic enrichi depuis fact_articles (fallback intelligent)
        - LEFT JOIN conservé
        - month inchangé
    """
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()

        # Résoudre le mois cible
        if month is not None:
            target_month = month
        else:
            result = conn.execute(
                "SELECT MAX(month) FROM fact_trend_reports WHERE category = ?",
                [category],
            ).fetchone()
            target_month = result[0] if result and result[0] else None

        if target_month is None:
            return []

        rows = conn.execute(
            """SELECT
                   t.topic_id,

                   COALESCE(
                       ANY_VALUE(a2.topic_label),
                       t.topic_label,
                       'Topic ' || t.topic_id
                   ) AS topic_label,

                   t.n_months_active,
                   tr.status,
                   tr.composite_score,
                   tr.trend_score,
                   tr.weak_signal_score,
                   tr.semantic_score,
                   tr.z_score_last,
                   tr.ols_slope,
                   tr.cagr,
                   tr.emergence_score_last,
                   tr.persistence_score,
                   tr.semantic_drift_last,
                   tr.keyword_drift_last,
                   tr.mean_novelty,
                   tr.mean_topic_prob,
                   tr.mean_centroid_distance,
                   tr.pct_english,
                   tr.mean_flesch_grade,
                   tr.mean_latex_density,
                   tr.pct_cross_category,
                   COALESCE(a.n_articles, 0) AS n_articles

               FROM dim_topics t

               LEFT JOIN fact_trend_reports tr
                   ON t.topic_id = tr.topic_id
                  AND t.category = tr.category
                  AND tr.month = ?

               LEFT JOIN (
                   SELECT topic_id, COUNT(*) AS n_articles
                   FROM fact_articles
                   WHERE category = ?
                   GROUP BY topic_id
               ) a ON t.topic_id = a.topic_id

               LEFT JOIN fact_articles a2
                   ON t.topic_id = a2.topic_id
                  AND t.category = a2.category

               WHERE t.category = ?
                 AND t.topic_id >= 0

               GROUP BY
                   t.topic_id,
                   t.topic_label,
                   t.n_months_active,
                   tr.status,
                   tr.composite_score,
                   tr.trend_score,
                   tr.weak_signal_score,
                   tr.semantic_score,
                   tr.z_score_last,
                   tr.ols_slope,
                   tr.cagr,
                   tr.emergence_score_last,
                   tr.persistence_score,
                   tr.semantic_drift_last,
                   tr.keyword_drift_last,
                   tr.mean_novelty,
                   tr.mean_topic_prob,
                   tr.mean_centroid_distance,
                   tr.pct_english,
                   tr.mean_flesch_grade,
                   tr.mean_latex_density,
                   tr.pct_cross_category,
                   a.n_articles

               ORDER BY COALESCE(tr.composite_score, -999) DESC
            """,
            [target_month, category, category],
        ).fetchall()

        def _f(v):
            return round(float(v), 4) if v is not None else None

        return [
            {
                "topic_id": int(r[0]),
                "topic_label": r[1] or f"Topic {r[0]}",
                "n_months_active": int(r[2]),
                "status": r[3],
                "composite_score": _f(r[4]),
                "trend_score": _f(r[5]),
                "weak_signal_score": _f(r[6]),
                "semantic_score": _f(r[7]),
                "z_score_last": _f(r[8]),
                "ols_slope": _f(r[9]),
                "cagr": _f(r[10]),
                "emergence_score_last": _f(r[11]),
                "persistence_score": _f(r[12]),
                "semantic_drift_last": _f(r[13]),
                "keyword_drift_last": _f(r[14]),
                "mean_novelty": _f(r[15]),
                "mean_topic_prob": _f(r[16]),
                "mean_centroid_distance": _f(r[17]),
                "pct_english": round(float(r[18]), 1) if r[18] is not None else None,
                "mean_flesch_grade": (
                    round(float(r[19]), 2) if r[19] is not None else None
                ),
                "mean_latex_density": _f(r[20]),
                "pct_cross_category": (
                    round(float(r[21]), 1) if r[21] is not None else None
                ),
                "n_articles": int(r[22]),
            }
            for r in rows
        ]

    except Exception as exc:
        logger.error("get_trends_overview '%s' échoué : %s", category, exc)
        return []
    finally:
        _close(conn)


def get_trend_history(category: str, topic_id: int) -> list[dict]:
    if not db_exists():
        return []
    conn = None
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT month, composite_score, trend_score, status
               FROM fact_trend_reports
               WHERE category = ? AND topic_id = ?
               ORDER BY month""",
            [category, topic_id],
        ).fetchall()
        return [
            {
                "month": r[0],
                "composite_score": round(float(r[1]), 4) if r[1] is not None else None,
                "trend_score": round(float(r[2]), 4) if r[2] is not None else None,
                "status": r[3],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("get_trend_history '%s'/%s échoué : %s", category, topic_id, exc)
        return []
    finally:
        _close(conn)


# --------------------------------------------------------------------------- #
# Suppression                                                                  #
# --------------------------------------------------------------------------- #


def delete_category_all(category: str) -> tuple[bool, str]:
    if not db_exists():
        return False, "dashboard.duckdb introuvable."
    conn = None
    try:
        from duckdb_sync.schema import delete_category

        conn = _connect(read_only=False)
        conn.execute("BEGIN")
        delete_category(conn, category)
        conn.execute("DELETE FROM dim_categories WHERE category = ?", [category])
        conn.execute("COMMIT")
        conn.close()
        return True, f"Catégorie '{category}' supprimée de DuckDB."
    except Exception as exc:
        logger.error("delete_category_all '%s' échoué : %s", category, exc)
        if conn is not None:
            try:
                conn.execute("ROLLBACK")
                conn.close()
            except Exception:
                pass
        return False, f"Erreur lors de la suppression : {exc}"
