"""
Rapport de résultat d'un run de traitement RAW → CLEAN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ProcessorReport:
    """
    Rapport complet d'un run de traitement arXiv RAW → CLEAN.
    """

    # Statut global
    status: str = "success"
    """Résultat global : 'success' | 'partial' | 'empty' | 'error'"""

    # Timing
    run_start: str | None = None
    """Timestamp ISO UTC du début du traitement."""

    run_end: str | None = None
    """Timestamp ISO UTC de la fin du traitement."""

    duration_seconds: float = 0.0
    """Durée totale du run en secondes."""

    # Métriques
    files_processed: list[str] = field(default_factory=list)
    """Chemins des fichiers Parquet CLEAN produits."""

    total_raw: int = 0
    """Nombre total d'articles lus depuis la zone RAW."""

    total_clean: int = 0
    """Nombre total d'articles écrits en zone CLEAN (après filtrage)."""

    total_dropped_invalid: int = 0
    """Articles rejetés pour schéma invalide (champ requis absent ou vide)."""

    total_dropped_duplicate: int = 0
    """Articles rejetés pour doublon (même arxiv_id_short dans un même fichier)."""

    # Erreurs
    errors: list[str] = field(default_factory=list)
    """Erreurs rencontrées (un fichier en erreur n'arrête pas les suivants)."""

    # Traçabilité
    config_snapshot: dict = field(default_factory=dict)
    """Snapshot de la configuration ProcessorConfig utilisée pour ce run."""

    @classmethod
    def build(
        cls,
        start: datetime,
        end: datetime,
        files_processed: list[str],
        total_raw: int,
        total_clean: int,
        total_dropped_invalid: int,
        total_dropped_duplicate: int,
        errors: list[str],
        config_snapshot: dict,
    ) -> "ProcessorReport":
        """
        Factory method : construit un ProcessorReport à partir des métriques d'un run.
        """
        duration = (end - start).total_seconds()

        if errors and not files_processed:
            status = "error"
        elif not files_processed and not errors:
            status = "empty"
        elif errors:
            status = "partial"
        else:
            status = "success"

        return cls(
            status=status,
            run_start=start.isoformat(),
            run_end=end.isoformat(),
            duration_seconds=round(duration, 2),
            files_processed=files_processed,
            total_raw=total_raw,
            total_clean=total_clean,
            total_dropped_invalid=total_dropped_invalid,
            total_dropped_duplicate=total_dropped_duplicate,
            errors=errors,
            config_snapshot=config_snapshot,
        )

    def summary_lines(self) -> list[str]:
        """Retourne les lignes de résumé pour les logs."""
        total_dropped = self.total_dropped_invalid + self.total_dropped_duplicate
        drop_rate = (
            f"{total_dropped / self.total_raw * 100:.1f}%"
            if self.total_raw > 0
            else "N/A"
        )
        return [
            f"Statut              : {self.status.upper()}",
            f"Fichiers CLEAN      : {len(self.files_processed)}",
            f"Articles RAW lus    : {self.total_raw}",
            f"Articles CLEAN écrits: {self.total_clean}",
            f"Rejetés (invalides) : {self.total_dropped_invalid}",
            f"Rejetés (doublons)  : {self.total_dropped_duplicate}",
            f"Taux de rejet       : {drop_rate}",
            f"Durée               : {self.duration_seconds:.2f}s",
        ]

    def to_json(self) -> str:
        """Sérialise le rapport en JSON indenté (UTF-8)."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: Path) -> None:
        """
        Sauvegarde le rapport JSON sur le disque.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
