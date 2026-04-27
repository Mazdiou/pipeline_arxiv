"""
Centraliser toute la configuration du processeur.

Pour ajouter un paramètre :
    1. L'ajouter dans ProcessorConfig avec sa valeur par défaut et sa docstring
    2. L'utiliser dans le module concerné via self.config.<param>
    3. Ajouter la validation nécessaire dans _validate()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field


@dataclass
class ProcessorConfig:
    """
    Configuration complète d'un run de traitement RAW to CLEAN.
    """

    # Chemins (utilisés uniquement avec storage_backend='local')
    raw_data_path: str = "data/RAW"
    """
    Chemin racine de la zone RAW du Data Lake (local).
    Ignoré si storage_backend='minio'.
    """

    clean_data_path: str = "data/CLEAN"
    """
    Chemin racine de la zone CLEAN du Data Lake (local).
    Ignoré si storage_backend='minio'.
    """

    # Filtres optionnels
    category: str = ""
    """
    Catégorie arXiv à traiter (ex: 'stat_ML', 'cs_AI').
    Vide = toutes les catégories trouvées dans la zone RAW.
    """

    year: int | None = None
    """Année à traiter (ex: 2026). None = toutes les années disponibles."""

    month: int | None = None
    """
    Mois à traiter (1-12). None = tous les mois disponibles.
    Requiert year si fourni.
    """

    # Comportement
    overwrite_existing: bool = False
    """
    Comportement si le fichier CLEAN cible existe déjà.
    True  : écrase le fichier existant.
    False : ignore le fichier et passe au suivant (défaut).
    """

    # Logs
    log_dir: str = "logs"
    """Dossier de stockage des fichiers de log horodatés."""

    # Backend de stockage
    storage_backend: str = field(
        default_factory=lambda: os.getenv("STORAGE_BACKEND", "local")
    )
    """
    Backend de stockage à utiliser.
    'local' : filesystem local 
    'minio' : serveur MinIO/S3
    """

    def __post_init__(self) -> None:
        """Déclenche la validation dès l'instanciation (fail-fast)."""
        self._validate()

    def _validate(self) -> None:
        """
        Vérifie la cohérence des paramètres.
        """
        if self.month is not None and self.year is None:
            raise ValueError("Paramètre 'year' requis quand 'month' est fourni.")

        if self.month is not None and not (1 <= self.month <= 12):
            raise ValueError(f"'month' doit être entre 1 et 12 (reçu : {self.month})")

        if self.year is not None and self.year < 1991:
            raise ValueError(
                f"'year' invalide : {self.year}. arXiv existe depuis 1991."
            )

        if self.storage_backend not in ("local", "minio"):
            raise ValueError(
                f"storage_backend invalide : '{self.storage_backend}'. "
                f"Valeurs acceptées : 'local', 'minio'"
            )

    def to_dict(self) -> dict:
        """Sérialise la configuration en dictionnaire."""
        return asdict(self)

    def summary(self) -> str:
        """Résumé lisible de la configuration pour les logs."""
        parts = []

        if self.category:
            parts.append(f"category='{self.category}'")
        else:
            parts.append("category=ALL")

        if self.year and self.month:
            parts.append(f"période={self.year}-{self.month:02d}")
        elif self.year:
            parts.append(f"année={self.year}")
        else:
            parts.append("période=ALL")

        parts.append(f"overwrite={self.overwrite_existing}")
        parts.append(f"backend={self.storage_backend}")
        return " | ".join(parts)
