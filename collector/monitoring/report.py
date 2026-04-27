"""
Le rapport est produit automatiquement à la fin de chaque ArxivCollector.run().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from ..config.constants import TOP_N_CATEGORIES


@dataclass
class RunReport:

    # Identifiant unique : lie le log, le rapport et l'entrée run_history
    run_id: str
    """Identifiant unique du run. Format : 'YYYYMMDD_HHMM_safe_category'."""

    # Statut du run
    status: str
    """'success' | 'partial' | 'empty' | 'error'

    - 'success' : articles collectés, aucune erreur
    - 'partial' : articles collectés malgré une erreur non-fatale
    - 'empty'   : aucun article dans la fenêtre temporelle, aucune erreur
    - 'error'   : échec sans article collecté
    """

    # Timing
    run_start: str
    """Timestamp ISO 8601 de début du run."""

    run_end: str
    """Timestamp ISO 8601 de fin du run."""

    duration_seconds: float
    """Durée totale du run en secondes, arrondie à 2 décimales."""

    # Résultats
    total_collected: int
    """Nombre d'articles effectivement collectés et sérialisés."""

    date_min: str | None
    """Date de publication la plus ancienne parmi les articles collectés (YYYY-MM-DD)."""

    date_max: str | None
    """Date de publication la plus récente parmi les articles collectés (YYYY-MM-DD)."""

    categories_distribution: dict[str, int]
    """Dictionnaire {catégorie: nombre d'articles}, limité au (TOP_N_CATEGORIES), trié par fréquence."""

    # Infrastructure
    output_path: str | None
    """Chemin absolu du dernier fichier articles.json sauvegardé.
    None si aucun article n'a été stocké (empty ou error)."""

    config_snapshot: dict
    """Snapshot complet de la CollectorConfig utilisée (résultat de to_dict())."""

    # Optionnel
    error_message: str | None = None
    """Message d'erreur si status == 'error' ou 'partial'. None sinon."""

    # Factory method
    @classmethod
    def build(
        cls,
        run_start: datetime,
        articles: list[dict],
        output_path: str | None,
        config_snapshot: dict,
        run_id: str = "",
        error_message: str | None = None,
    ) -> "RunReport":
        """
        Construit un RunReport à partir des données brutes du run.

        Args:
            run_start       : Datetime de début du run (datetime.utcnow()).
            articles        : Liste des articles collectés (peut être vide).
            output_path     : Chemin du fichier sauvegardé (None si aucun).
            config_snapshot : Config sérialisée (CollectorConfig.to_dict()).
            run_id          : Identifiant unique du run (depuis ArxivCollector.run_id).
            error_message   : Message d'erreur éventuel (None si pas d'erreur).

        Returns:
            Instance RunReport prête à être loggée, sérialisée ou sauvegardée.
        """
        run_end = datetime.now(timezone.utc)
        duration = (run_end - run_start).total_seconds()

        # Dates de publication des articles
        pub_dates = [a["published_date"] for a in articles if a.get("published_date")]

        # Distribution des catégories (toutes catégories, pas seulement la primaire)
        categories: dict[str, int] = {}
        for article in articles:
            for cat in article.get("categories") or []:
                categories[cat] = categories.get(cat, 0) + 1

        # Top N catégories triées par fréquence décroissante
        top_categories = dict(
            sorted(categories.items(), key=lambda x: -x[1])[:TOP_N_CATEGORIES]
        )

        # Calcul du statut
        if error_message:
            status = "partial" if articles else "error"
        else:
            status = "success" if articles else "empty"

        return cls(
            run_id=run_id,
            status=status,
            run_start=run_start.isoformat(),
            run_end=run_end.isoformat(),
            duration_seconds=round(duration, 2),
            total_collected=len(articles),
            date_min=min(pub_dates) if len(articles) > 0 else None,
            date_max=max(pub_dates) if len(articles) > 0 else None,
            categories_distribution=top_categories,
            output_path=output_path,
            config_snapshot=config_snapshot,
            error_message=error_message,
        )

    # Sérialisation
    def to_dict(self) -> dict:
        """
        Sérialise le rapport en dictionnaire Python.
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """
        Sérialise le rapport en chaîne JSON.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: Path) -> None:
        """
        Sauvegarde le rapport dans un fichier JSON.

        Raises:
            IOError / OSError : si l'écriture échoue (permissions, espace disque).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    # Affichage
    def summary_lines(self) -> list[str]:
        """
        Retourne une liste de lignes résumant le rapport pour les logs.
        """
        lines = [
            "=" * 60,
            "RAPPORT DE RUN",
            f"  Run ID           : {self.run_id}",
            f"  Statut           : {self.status.upper()}",
            f"  Démarrage        : {self.run_start}",
            f"  Fin              : {self.run_end}",
            f"  Durée            : {self.duration_seconds}s",
            f"  Articles         : {self.total_collected}",
            f"  Période couverte : {self.date_min} → {self.date_max}",
            f"  Fichier RAW      : {self.output_path}",
        ]

        # Distribution des catégories (top 5, TOP_N_CATEGORIES)
        if self.categories_distribution:
            lines.append("  Top catégories   :")
            for cat, count in self.categories_distribution.items():
                lines.append(f"    - {cat}: {count}")

        # Ligne d'erreur ajoutée uniquement si présente
        if self.error_message:
            lines.append(f"  Erreur           : {self.error_message}")

        lines.append("=" * 60)
        return lines
