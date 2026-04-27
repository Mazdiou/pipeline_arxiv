"""
Organisation par domaine :
  API arXiv    : limites et contraintes imposées par l'API arXiv
  Backoff HTTP : stratégie de gestion du rate limit (429/503/504)
  Collecteur   : comportement opérationnel (seuils, logs)
  Validation   : ensembles de valeurs autorisées pour la config
  Schéma       : version et contrat de sortie du sérialiseur
  Rapport      : paramètres de mise en forme du RunReport
  Fichiers     : noms des fichiers persistants du pipeline
"""

from __future__ import annotations

# API arXiv
ARXIV_MAX_BATCH_SIZE: int = 2000
"""Nombre maximum de résultats par page imposé par l'API arXiv."""

ARXIV_RECOMMENDED_BATCH_SIZE: int = 300
"""Nombre recommandé de résultats par page imposé par l'API arXiv."""

ARXIV_MIN_DELAY_SEC: int = 4
"""
Délai minimum en secondes entre les appels API arXiv.
En dessous de ce seuil, le risque d'obtenir un HTTP 429 est élevé.
Aligné sur la politeness policy officielle d'arXiv.
"""

# Backoff HTTP 429 / 503 / 504
BACKOFF_INITIAL_DELAY: int = 60
"""Délai initial en secondes après la première erreur HTTP retriable."""

BACKOFF_MULTIPLIER: int = 2
"""Facteur multiplicatif appliqué au délai à chaque nouvelle tentative."""

BACKOFF_MAX_RETRIES: int = 4
"""Nombre maximum de tentatives de reprise avant abandon définitif."""

BACKOFF_MAX_DELAY: int = 600
"""Délai maximum entre deux tentatives (10 min) pour éviter des attentes excessives."""

# Comportement opérationnel
EARLY_STOP_THRESHOLD: int = 50
"""
Nombre d'articles consécutifs hors-fenêtre déclenchant l'arrêt anticipé.
Exploite le tri décroissant de l'API arXiv : dès que N articles sont trop anciens,
tous les suivants le seront aussi.
"""

LOG_PROGRESS_EVERY: int = 300
"""Fréquence d'affichage de la progression dans les logs."""

# Validation de la configuration
VALID_SORT_BY: frozenset[str] = frozenset(
    {"submittedDate", "relevance", "lastUpdatedDate"}
)
"""Critères de tri acceptés par l'API arXiv."""

VALID_SORT_ORDER: frozenset[str] = frozenset({"descending", "ascending"})
"""Ordres de tri acceptés par l'API arXiv."""


REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "arxiv_id",
        "title",
        "abstract",
        "published",
        "authors",
        "_collected_at",
    }
)
"""
Champs garantis présents et non-None dans chaque article sérialisé.
Toute violation lève une ValueError immédiate (fail-fast).
"""

# Rapport de run
TOP_N_CATEGORIES: int = 5
"""Nombre de catégories à inclure dans le top du RunReport."""

# Fichiers persistants
HISTORY_FILENAME: str = "run_history.json"
"""Nom du fichier d'historique des runs, stocké dans base_path."""

# Semantic Scholar API
SS_BATCH_ENDPOINT: str = "https://api.semanticscholar.org/graph/v1/paper/batch"
"""Endpoint batch officiel pour récupérer les métadonnées de plusieurs articles."""

SS_MAX_BATCH_SIZE: int = 500
"""Taille de lot maximum acceptée par l'API Semantic Scholar."""

SS_REQUEST_DELAY: float = 1.0
"""Délai en secondes entre deux lots consécutifs avec clé API (100 req/s autorisé)."""

SS_REQUEST_DELAY_NO_KEY: float = 5.0
"""
Délai en secondes entre deux lots sans clé API.
Sans clé, la limite est ~1 req/3s — on cale dessus sans marge excessive.
"""

SS_TIMEOUT: int = 30
"""Timeout en secondes pour chaque requête HTTP vers Semantic Scholar."""

SS_BACKOFF_INITIAL_DELAY: int = 60
"""Délai initial en secondes après un HTTP 429 de Semantic Scholar."""

SS_BACKOFF_MAX_RETRIES: int = 3
"""Nombre maximum de tentatives par lot avant abandon définitif."""

SS_BACKOFF_MAX_DELAY: int = 300
"""Délai maximum entre deux tentatives (5 min)."""

ARXIV_PAGINATION_WALL: int = 9500
"""
Offset à partir duquel arXiv retourne HTTP 500 (limite dure ~10 000 résultats).
En dessous de ce seuil, un 500 est une erreur inattendue. Au-dessus, c'est le mur
de pagination : on stoppe proprement sans lever d'exception.
"""

_ERROR_LABELS: dict[int, str] = {
    429: "Rate Limit (429)",
    503: "Service Unavailable (503)",
    504: "Gateway Timeout (504)",
}
"""Libellés lisibles pour les codes d'erreur HTTP gérés par le backoff."""
