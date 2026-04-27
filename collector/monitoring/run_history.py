"""
Suivi de l'historique des runs de collecte.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..config.constants import HISTORY_FILENAME

if TYPE_CHECKING:
    from .report import RunReport


class RunHistoryMonitor:
    """
    Gestionnaire de l'historique des runs de collecte.
    """

    def __init__(
        self,
        base_path: str = "logs/history",
        logger: logging.Logger | None = None,
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger("arxiv.collector.monitoring")
        self.history_file = self.base_path / HISTORY_FILENAME

    # Interface publique
    def record(self, report: "RunReport", config_snapshot: dict) -> None:
        """
        Enregistre un run terminé dans l'historique.

        Args:
            report          : RunReport du run terminé
            config_snapshot : Snapshot de la config utilisée
        """
        history = self._load()

        run_entry = {
            "run_id": report.run_id,
            "status": report.status,
            "category": config_snapshot.get("category", ""),
            "query": config_snapshot.get("query", ""),
            "period_start": report.date_min,
            "period_end": report.date_max,
            "lookback_days": config_snapshot.get("lookback_days"),
            "total_articles": report.total_collected,
            "categories_distribution": report.categories_distribution,
            "output_path": report.output_path,
            "duration_s": report.duration_seconds,
            "run_at": report.run_start,
            "error": report.error_message,
        }

        history["runs"].append(run_entry)
        history["_last_updated"] = datetime.now(timezone.utc).isoformat()

        self._save(history)
        self.logger.info(f"Run enregistré dans l'historique → {self.history_file}")

    # I/O
    def _load(self) -> dict:
        """Charge l'historique depuis le fichier JSON. Crée un historique vide si absent."""
        if not self.history_file.exists():
            return {"runs": [], "_last_updated": None}
        try:
            return json.loads(self.history_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.warning(f"Historique illisible ({exc}), réinitialisation.")
            return {"runs": [], "_last_updated": None}

    def _save(self, history: dict) -> None:
        """Sauvegarde l'historique dans le fichier JSON de façon atomique."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        content = json.dumps(history, ensure_ascii=False, indent=2)
        tmp_path = self.history_file.with_suffix(".tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, self.history_file)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
