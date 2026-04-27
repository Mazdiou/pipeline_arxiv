"""
Ce fichier contient uniquement le dataclass CollectorConfig.

Pour ajouter un paramètre :
    1. L'ajouter dans CollectorConfig avec sa valeur par défaut et sa docstring
    2. L'utiliser dans le module concerné via self.config.<param>
    3. Ajouter la validation nécessaire dans _validate()
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .constants import (
    ARXIV_MAX_BATCH_SIZE,
    ARXIV_MIN_DELAY_SEC,
    VALID_SORT_BY,
    VALID_SORT_ORDER,
    ARXIV_RECOMMENDED_BATCH_SIZE,
    SS_MAX_BATCH_SIZE,
)


@dataclass
class CollectorConfig:
    """
    Configuration complète d'un run de collecte arXiv.

    Tous les champs sont documentés, typés et validés automatiquement
    à l'instanciation via __post_init__()

    Les valeurs par défaut permettent un usage immédiat avec un minimum
    de paramètres (seul category ou query est obligatoire).
    """

    # Thématique
    category: str = ""
    """
        Catégorie arXiv ciblée (ex: 'cs.AI', 'stat.ML', 'q-bio.NC').
        Voir : https://arxiv.org/category_taxonomy
    """

    query: str = ""
    """
        Mots-clés libres complémentaires (ex: 'large language models').
        Combinée avec category via AND si les deux sont renseignés.
    """

    # Fenêtre temporelle
    lookback_days: int = 180
    """
        Nombre de jours en arrière à collecter depuis aujourd'hui (UTC).
        Ignoré si start_date est fourni (start_date est prioritaire).
    """

    start_date: str | None = None
    """
        Date de début au format 'YYYY-MM-DD' (UTC).
        Prioritaire sur lookback_days si fournie.
    """

    end_date: str | None = None
    """
        Date de fin au format 'YYYY-MM-DD' (UTC).
        Défaut : aujourd'hui (UTC) si non fournie.
    """

    # Limites de collecte
    max_results: int = 9000
    """Nombre maximum d'articles à retourner par run."""

    batch_size: int = min(ARXIV_RECOMMENDED_BATCH_SIZE, ARXIV_MAX_BATCH_SIZE)
    """
        Taille des lots pour la pagination arXiv.
    """

    # Comportement API
    delay_between_batches: int = ARXIV_MIN_DELAY_SEC
    """
        Délai en secondes entre les appels API.
    """

    # Tri
    sort_by: str = "submittedDate"
    """
        Critère de tri des résultats arXiv.
        Valeurs acceptées : ["submittedDate", "relevance", "lastUpdatedDate"].
    """

    sort_order: str = "descending"
    """
        Ordre de tri des résultats.
        Valeurs acceptées : ["descending", "ascending"].
    """

    # Stockage
    raw_data_path: str = "data/RAW"
    """
        Chemin racine de la zone RAW du Data Lake.
    """

    overwrite_existing: bool = False
    """
        Comportement en cas de fichier existant.
        True  : écrase le fichier existant.
        False : fusionne avec déduplication par arxiv_id_short.
    """

    # Logs
    log_dir: str = "logs"
    """Dossier de stockage des fichiers de log horodatés."""

    # Citations Semantic Scholar
    fetch_citations: bool = True
    """
        Active l'enrichissement des citations depuis Semantic Scholar après la collecte.
    """

    citations_batch_size: int = 500
    """
        Nombre d'articles par lot pour l'API Semantic Scholar.
        500 est la limite officielle du batch endpoint (confirmée dans le swagger).
    """

    citations_api_key: str | None = None
    """
        Clé API Semantic Scholar.
        Obtenir une clé : https://www.semanticscholar.org/product/api
    """

    # Validation automatique
    def __post_init__(self) -> None:
        """Déclenche la validation dès l'instanciation (fail-fast)."""
        self._validate()

    def _validate(self) -> None:
        """
        Vérifie la cohérence de tous les paramètres.

        Raises:
            ValueError : si un paramètre est invalide ou incohérent.
        """
        # Au moins category ou query doit être fourni
        if not self.category and not self.query:
            raise ValueError("Au moins 'category' ou 'query' doit être fourni.")

        # Critère de tri valide
        if self.sort_by not in VALID_SORT_BY:
            raise ValueError(
                f"sort_by invalide : '{self.sort_by}'. "
                f"Valeurs acceptées : {sorted(VALID_SORT_BY)}"
            )

        # Ordre de tri valide
        if self.sort_order not in VALID_SORT_ORDER:
            raise ValueError(
                f"sort_order invalide : '{self.sort_order}'. "
                f"Valeurs acceptées : {sorted(VALID_SORT_ORDER)}"
            )

        # max_results strictement positif
        if self.max_results <= 0:
            raise ValueError(f"max_results doit être > 0 (reçu : {self.max_results})")

        # batch_size strictement positif
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être > 0 (reçu : {self.batch_size})")

        # batch_size dans la limite arXiv
        if self.batch_size > ARXIV_MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size doit être ≤ {ARXIV_MAX_BATCH_SIZE} (limite arXiv). "
            )

        # Délai non-négatif
        if self.delay_between_batches < 0:
            raise ValueError(f"delay_between_batches doit être ≥ 0 ")

        # Délai minimum recommandé : warning uniquement
        if self.delay_between_batches < ARXIV_MIN_DELAY_SEC:
            import warnings

            warnings.warn(
                f"delay_between_batches ({self.delay_between_batches}s) "
                f"< minimum recommandé ({ARXIV_MIN_DELAY_SEC}s). "
                f"Sera ajusté automatiquement à la valeur recommandée.",
                stacklevel=2,
            )

        # lookback_days strictement positif
        if self.lookback_days <= 0:
            raise ValueError(
                f"lookback_days doit être > 0 (reçu : {self.lookback_days})"
            )

        # citations_batch_size dans la limite Semantic Scholar
        if (
            self.citations_batch_size <= 0
            or self.citations_batch_size > SS_MAX_BATCH_SIZE
        ):
            raise ValueError(
                f"citations_batch_size doit être entre 1 et {SS_MAX_BATCH_SIZE} "
                f"(reçu : {self.citations_batch_size})"
            )

    # Sérialisation
    def to_dict(self) -> dict:
        """
        Sérialise la configuration en dictionnaire.
        """
        return asdict(self)

    def summary(self) -> str:
        """
        Retourne un résumé de la configuration pour les logs ou rapports.
        """

        if self.start_date:
            end = self.end_date or "aujourd'hui"
            period = f"du {self.start_date} au {end}"
        else:
            period = f"derniers {self.lookback_days} jours"

        return (
            f"category={self.category or 'N/A'}, "
            f"query={self.query or 'N/A'}, période={period}, "
            f"max_results={self.max_results}, tri={self.sort_by} ({self.sort_order}) "
        )
