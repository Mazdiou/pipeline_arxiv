"""
config.py

Configuration MinIO partagée entre les modules collector, processor et enricher.

Lit les paramètres depuis le fichier .env (ou variables d'environnement système).
Expose MINIO_CONFIG (singleton) consommé uniquement par les minio_storage.py
de chaque module — jamais dans les settings.py des modules.

Variables .env attendues :
    MINIO_ENDPOINT      : adresse du serveur MinIO
    MINIO_ACCESS_KEY    : clé d'accès
    MINIO_SECRET_KEY    : clé secrète
    MINIO_RAW_BUCKET    : bucket zone RAW
    MINIO_CLEAN_BUCKET  : bucket zone CLEAN
    MINIO_GOLD_BUCKET   : bucket zone GOLD
    MINIO_TRENDS_BUCKET : bucket zone TRENDS
    MINIO_SECURE        : HTTPS (true/false)
    MINIO_OVERWRITE     : écraser les objets
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Charger le .env si python-dotenv est disponible
try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass  # Sans dotenv → variables d'environnement système utilisées directement


@dataclass(frozen=True)
class MinioConfig:
    """Paramètres de connexion et de configuration MinIO."""

    endpoint: str
    access_key: str
    secret_key: str
    raw_bucket: str
    clean_bucket: str
    gold_bucket: str
    trends_bucket: str
    secure: bool
    overwrite: bool


MINIO_CONFIG = MinioConfig(
    endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "password123"),
    raw_bucket=os.getenv("MINIO_RAW_BUCKET", "arxiv-raw"),
    clean_bucket=os.getenv("MINIO_CLEAN_BUCKET", "arxiv-clean"),
    gold_bucket=os.getenv("MINIO_GOLD_BUCKET", "arxiv-gold"),
    trends_bucket=os.getenv("MINIO_TRENDS_BUCKET", "arxiv-trends"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    overwrite=os.getenv("MINIO_OVERWRITE", "false").lower() == "true",
)
