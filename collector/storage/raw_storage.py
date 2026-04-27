"""
Backend de stockage RAW (filesystem) pour la zone RAW.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from .base_storage import BaseStorage, _group_by_month, _merge, _safe_category_name


# Classe RawStorage : implémentation locale du stockage RAW
class RawStorage(BaseStorage):
    """
    Gestion du stockage RAW sur le filesystem local.

    Structure générée :
    {base_path}/
    └── {safe_category}/
        └── {YYYY}/
            └── {MM}/
                └── articles.json
    """

    def __init__(
        self,
        base_path: str = "data/RAW",
        overwrite: bool = False,
        logger: logging.Logger | None = None,
    ):
        """
        Initialise le stockage RAW.

        Args:
            base_path : Chemin racine de la zone RAW
            overwrite : True = écrase les fichiers existants
                        False = fusionne avec déduplication
            logger    : Logger applicatif
        """
        self.base_path = Path(base_path)
        self.overwrite = overwrite
        self.logger = logger or logging.getLogger("arxiv.collector.storage")

    # Interface publique
    def save(self, articles: list[dict], category: str, run_date: datetime) -> str:
        """
        Sauvegarde les articles dans la zone RAW.

        Args:
            articles : Liste de dictionnaires d'articles sérialisés
            category : Catégorie arXiv (ex: 'cs.AI')
            run_date : Date du run (utilisée en fallback si pas de date de publication)

        Returns:
            Chemin du fichier sauvegardé (str)
        """
        # Regroupement des articles par mois de publication
        # un fichier par mois : data/RAW/cs_AI/2024/01/articles.json
        articles_by_month = _group_by_month(articles, run_date, self.logger)
        last_path: Path | None = None

        for month_dt, month_articles in articles_by_month.items():
            output_path = self._build_path(category, month_dt)

            # Fusion ou remplacement selon le mode overwrite
            if self.overwrite or not output_path.exists():
                final_articles = month_articles
                new_count = len(month_articles)
                dup_count = 0
                updated_count = 0
            else:
                existing = self._load_existing(output_path)
                final_articles, new_count, dup_count, updated_count = _merge(
                    existing, month_articles, self.logger
                )

            self.logger.info(
                f"[{month_dt.strftime('%Y-%m')}] RAW Storage | "
                f"New: {new_count} | Updated: {updated_count} | Duplicates: {dup_count} | "
                f"File: {output_path}"
            )

            self._write(output_path, final_articles, new_count, updated_count)
            last_path = output_path

        # Retourne le dernier fichier écrit
        return str(last_path or self._build_path(category, run_date))

    def load(self, path: str) -> list[dict]:
        """
        Charge les articles depuis un fichier RAW existant.

        Args:
            path : Chemin du fichier articles.json

        Returns:
            Liste de dictionnaires d'articles
        """
        return self._load_existing(Path(path))

    # Méthodes internes
    def _build_path(self, category: str, ref_date: datetime) -> Path:
        """
        Construit le chemin de fichier structuré par catégorie et mois de publication.

        Structure : {base_path}/{safe_category}/{YYYY}/{MM}/articles.json

        Args:
            category : Catégorie arXiv
            ref_date : Date de référence pour le mois (datetime)

        Returns:
            Path complet vers le fichier articles.json
        """
        month_subpath = ref_date.strftime("%Y/%m")
        dir_path = self.base_path / _safe_category_name(category) / month_subpath
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / "articles.json"

    def _load_existing(self, path: Path) -> list[dict]:
        """
        Lit et parse un fichier articles.json existant.

        Retourne une liste vide si le fichier est absent ou illisible.

        Returns:
            Liste d'articles existants
        """
        if not path.exists():
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing = data.get("articles", [])
            self.logger.debug(
                f"Fichier existant chargé : {len(existing)} articles : {path}"
            )
            return existing
        except (json.JSONDecodeError, KeyError) as exc:
            self.logger.warning(
                f"Fichier existant illisible ({exc}), écrasement prévu : {path}"
            )
            return []

    def _write(
        self, path: Path, articles: list[dict], new_count: int, updated_count: int = 0
    ) -> None:
        """
        Écrit le fichier JSON avec métadonnées de run de façon atomique.

        Args:
            path          : Chemin de destination
            articles      : Articles à écrire (liste complète après fusion)
            new_count     : Nombre de nouveaux articles ajoutés ce run
            updated_count : Nombre de versions mises à jour ce run

        Raises:
            IOError : si l'écriture ou la validation échoue
        """
        payload = {
            "_metadata": {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "total_articles": len(articles),
                "new_articles_this_run": new_count,
                "updated_articles_this_run": updated_count,
                "file_path": str(path),
            },
            "articles": articles,
        }

        tmp_path = path.with_suffix(".tmp")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp_path, path)  # écriture atomique
        except (IOError, OSError) as exc:
            tmp_path.unlink(missing_ok=True)
            self.logger.error(f"Erreur écriture RAW ({type(exc).__name__}): {exc}")
            raise IOError(
                f"Impossible d'écrire {path}. Vérifiez permissions et espace disque."
            ) from exc

        self.logger.info(
            f"Fichier RAW sauvegardé et validé : {path} ({len(articles)} articles)"
        )
