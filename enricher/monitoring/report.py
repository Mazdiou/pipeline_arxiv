"""
Rapport d'enrichissement produit par ArxivEnricher après chaque run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PartitionStats:
    """Statistiques NLP pour une partition GOLD (catégorie/année/mois)."""

    category: str
    year: int
    month: int

    # Compteurs articles
    n_articles: int = 0
    n_skipped: int = 0
    n_enriched: int = 0
    n_failed: int = 0

    # Topics
    n_topics: int = 0
    n_topic_outliers: int = 0
    topic_outlier_pct: float = 0.0

    # Langue
    n_non_english: int = 0
    non_english_pct: float = 0.0

    # Qualité
    n_short_abstract: int = 0

    # Novelty
    avg_novelty_score: float = 0.0
    avg_intra_similarity: float = 0.0

    # NER
    avg_entities_per_article: float = 0.0

    # Erreur éventuelle
    error: Optional[str] = None

    def to_log_lines(self) -> list[str]:
        """Formate les statistiques en lignes de log lisibles."""
        lines = [
            f"  Partition : {self.category}/{self.year}/{self.month:02d}",
            f"  Articles  : {self.n_articles} (enrichis={self.n_enriched}, "
            f"sautés={self.n_skipped}, erreurs={self.n_failed})",
            f"  Topics    : {self.n_topics} topics | outliers={self.n_topic_outliers} "
            f"({self.topic_outlier_pct:.1f}%)",
            f"  Langue    : {self.n_non_english} non-anglais "
            f"({self.non_english_pct:.1f}%)",
            f"  Qualité   : short={self.n_short_abstract}",
            f"  Novelty   : avg_novelty={self.avg_novelty_score:.3f} | "
            f"avg_sim={self.avg_intra_similarity:.3f}",
            f"  NER       : avg_entities={self.avg_entities_per_article:.1f}/article",
        ]
        if self.error:
            lines.append(f"  ERREUR    : {self.error}")
        return lines


@dataclass
class EnrichmentReport:
    """
    Rapport complet d'un run ArxivEnricher.

    Agrège les PartitionStats de toutes les partitions traitées.
    """

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    partitions: list[PartitionStats] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        """Nombre total de partitions traitées."""
        return len(self.partitions)

    @property
    def n_success(self) -> int:
        """Nombre de partitions enrichies sans erreur."""
        return sum(1 for p in self.partitions if p.error is None and p.n_enriched > 0)

    @property
    def n_errors(self) -> int:
        """Nombre de partitions en erreur."""
        return sum(1 for p in self.partitions if p.error is not None)

    @property
    def n_articles_total(self) -> int:
        """Nombre total d'articles enrichis."""
        return sum(p.n_enriched for p in self.partitions)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Durée totale du run en secondes."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def add_partition(self, stats: PartitionStats) -> None:
        """Ajoute les stats d'une partition au rapport."""
        self.partitions.append(stats)

    def finalize(self) -> None:
        """Marque la fin du run."""
        self.finished_at = datetime.now(timezone.utc)

    def summary_lines(self) -> list[str]:
        """Retourne un résumé compact du run pour les logs."""
        duration = self.duration_seconds
        duration_str = f"{duration:.1f}s" if duration is not None else "N/A"

        lines = [
            "=" * 60,
            "RAPPORT D'ENRICHISSEMENT",
            "=" * 60,
            f"Partitions : {self.n_total} traitées | "
            f"{self.n_success} succès | {self.n_errors} erreurs",
            f"Articles   : {self.n_articles_total} enrichis",
            f"Durée      : {duration_str}",
            "-" * 60,
        ]
        for stats in self.partitions:
            lines.extend(stats.to_log_lines())
            lines.append("")
        lines.append("=" * 60)
        return lines

    def log_summary(self, logger: logging.Logger) -> None:
        """Écrit le résumé complet dans le logger fourni."""
        for line in self.summary_lines():
            logger.info(line)
