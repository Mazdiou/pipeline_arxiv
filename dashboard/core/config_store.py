"""
dashboard/core/config_store.py

Persistance des configurations par catégorie dans pipeline_configs.json.

Séparé de DuckDB : la config est un artefact de déploiement,
pas une donnée analytique.

Structure JSON :
    {
      "stat_ML": {
        "display_name": "Statistics - Machine Learning",
        "query": "",
        "lookback_days": 180,
        "start_date": null,
        "end_date": null,
        "max_results": 6000,
        "fetch_citations": true,
        "sort_by": "submittedDate",
        "sort_order": "descending",
        "nr_topics": null,
        "min_months_for_trend": 4,
        "z_score_window": 6,
        "semantic_drift_enabled": true
      }
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("data/pipeline_configs.json")


@dataclass
class CategoryConfig:
    """
    Configuration persistante d'une catégorie surveillée.

    Fenêtre temporelle :
        - mode "lookback" : start_date=None, lookback_days utilisé par le collecteur
        - mode "period"   : start_date fourni (prioritaire sur lookback_days),
                            end_date optionnel (défaut = aujourd'hui)
    """

    display_name: str
    query: str = ""
    # Fenêtre temporelle
    lookback_days: int = 180           # jours en arrière (ignoré si start_date fourni)
    start_date: Optional[str] = None   # "YYYY-MM-DD" — prioritaire sur lookback_days
    end_date: Optional[str] = None     # "YYYY-MM-DD" — défaut aujourd'hui si None
    # Collecte
    max_results: int = 6000
    fetch_citations: bool = True
    sort_by: str = "submittedDate"     # submittedDate | relevance | lastUpdatedDate
    sort_order: str = "descending"     # descending | ascending
    # NLP — topics
    nr_topics: Optional[int] = None
    # NLP — tendances
    min_months_for_trend: int = 4
    z_score_window: int = 6
    semantic_drift_enabled: bool = True

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.display_name or not self.display_name.strip():
            raise ValueError("display_name ne peut pas être vide.")
        if self.lookback_days <= 0:
            raise ValueError("lookback_days doit être > 0.")
        if self.start_date is not None:
            try:
                start = date.fromisoformat(self.start_date)
            except ValueError:
                raise ValueError(
                    f"start_date invalide : '{self.start_date}'. Format attendu : YYYY-MM-DD."
                )
            if self.end_date is not None:
                try:
                    end = date.fromisoformat(self.end_date)
                except ValueError:
                    raise ValueError(
                        f"end_date invalide : '{self.end_date}'. Format attendu : YYYY-MM-DD."
                    )
                if end < start:
                    raise ValueError(
                        f"end_date ({self.end_date}) doit être >= start_date ({self.start_date})."
                    )
        elif self.end_date is not None:
            # end_date sans start_date n'a pas de sens
            raise ValueError("end_date ne peut pas être défini sans start_date.")
        if not (1000 <= self.max_results <= 9900):
            raise ValueError("max_results doit être entre 1 000 et 9 900.")
        valid_sort_by = {"submittedDate", "relevance", "lastUpdatedDate"}
        if self.sort_by not in valid_sort_by:
            raise ValueError(
                f"sort_by invalide : '{self.sort_by}'. Valeurs : {sorted(valid_sort_by)}."
            )
        if self.sort_order not in {"descending", "ascending"}:
            raise ValueError(
                f"sort_order invalide : '{self.sort_order}'. Valeurs : descending, ascending."
            )
        if self.nr_topics is not None and self.nr_topics < 2:
            raise ValueError("nr_topics doit être >= 2 (ou None pour auto).")
        if not (2 <= self.min_months_for_trend <= 12):
            raise ValueError("min_months_for_trend doit être entre 2 et 12.")
        if not (2 <= self.z_score_window <= 12):
            raise ValueError("z_score_window doit être entre 2 et 12.")

    # ------------------------------------------------------------------ #
    # Sérialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CategoryConfig":
        return cls(
            display_name=data.get("display_name", ""),
            query=data.get("query", ""),
            lookback_days=int(data.get("lookback_days", 180)),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            max_results=int(data.get("max_results", 6000)),
            fetch_citations=bool(data.get("fetch_citations", True)),
            sort_by=data.get("sort_by", "submittedDate"),
            sort_order=data.get("sort_order", "descending"),
            nr_topics=data.get("nr_topics"),
            min_months_for_trend=int(data.get("min_months_for_trend", 4)),
            z_score_window=int(data.get("z_score_window", 6)),
            semantic_drift_enabled=bool(data.get("semantic_drift_enabled", True)),
        )


class ConfigStore:
    """
    Gestionnaire R/W du fichier pipeline_configs.json.

    Chaque opération relit le fichier depuis le disque pour éviter
    les conflits de cache entre reruns Streamlit.

    Args:
        path : Chemin vers pipeline_configs.json.
                Défaut : data/pipeline_configs.json
    """

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self._path = path

    # ------------------------------------------------------------------ #
    # Lecture                                                              #
    # ------------------------------------------------------------------ #

    def load_all(self) -> dict[str, CategoryConfig]:
        """
        Retourne toutes les configs indexées par category_id.
        Retourne {} si le fichier est absent ou corrompu.
        """
        if not self._path.exists():
            return {}
        try:
            raw: dict = json.loads(self._path.read_text(encoding="utf-8"))
            result: dict[str, CategoryConfig] = {}
            for cat_id, data in raw.items():
                try:
                    result[cat_id] = CategoryConfig.from_dict(data)
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Config corrompue ignorée pour '%s' : %s", cat_id, exc
                    )
            return result
        except json.JSONDecodeError as exc:
            logger.error(
                "pipeline_configs.json illisible (JSON invalide) : %s", exc
            )
            return {}

    def get(self, category_id: str) -> Optional[CategoryConfig]:
        """Retourne la config d'une catégorie, ou None si absente."""
        return self.load_all().get(category_id)

    def exists(self, category_id: str) -> bool:
        """Vérifie si une catégorie est déjà configurée."""
        return category_id in self.load_all()

    def list_ids(self) -> list[str]:
        """Retourne la liste triée des category_id configurés."""
        return sorted(self.load_all().keys())

    # ------------------------------------------------------------------ #
    # Écriture                                                             #
    # ------------------------------------------------------------------ #

    def save_all(self, configs: dict[str, CategoryConfig]) -> None:
        """Écrase le fichier avec le dict fourni (opération atomique via tmp)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in configs.items()}
        tmp = self._path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self._path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise exc

    def upsert(self, category_id: str, config: CategoryConfig) -> None:
        """Insère ou met à jour la config d'une catégorie."""
        configs = self.load_all()
        configs[category_id] = config
        self.save_all(configs)

    def delete(self, category_id: str) -> bool:
        """
        Supprime la config d'une catégorie.

        Returns:
            True si supprimée, False si absente.
        """
        configs = self.load_all()
        if category_id not in configs:
            return False
        del configs[category_id]
        self.save_all(configs)
        return True


def normalize_category_id(arxiv_id: str) -> str:
    """
    Convertit un identifiant arXiv en clé interne du pipeline.

    Règle : remplace les points par des underscores, strip les espaces.
    Exemples :
        'stat.ML'  → 'stat_ML'
        'cs.AI'    → 'cs_AI'
        'stat_ML'  → 'stat_ML'  (déjà normalisé)

    Args:
        arxiv_id : Identifiant arXiv (ex: 'stat.ML').

    Returns:
        Clé interne (ex: 'stat_ML').
    """
    return arxiv_id.strip().replace(".", "_")


def validate_arxiv_id(arxiv_id: str) -> Optional[str]:
    """
    Valide le format d'un identifiant arXiv.

    Formats acceptés : 'primary.Sub', 'primary_Sub', 'primary.sub.Sub'.

    Args:
        arxiv_id : Identifiant brut saisi par l'utilisateur.

    Returns:
        Message d'erreur (str) si invalide, None si valide.
    """
    clean = arxiv_id.strip()
    if not clean:
        return "L'identifiant arXiv ne peut pas être vide."
    normalized = normalize_category_id(clean)
    if "_" not in normalized:
        return (
            "Format invalide. Exemple attendu : 'stat.ML', 'cs.AI', 'math.AP'."
        )
    parts = normalized.split("_", 1)
    if not all(p.replace("-", "").isalnum() for p in parts):
        return "L'identifiant ne doit contenir que des lettres, chiffres, tirets et points."
    return None
