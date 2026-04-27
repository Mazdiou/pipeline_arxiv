"""
Backends de stockage pour le pipeline de traitement arXiv RAW → CLEAN.

Exports :
    BaseProcessorStorage   : interface abstraite (ABC)
    RawPartition           : identifiant de partition (category/year/month)
    LocalProcessorStorage  : filesystem local (défaut)
    MinioProcessorStorage  : serveur MinIO/S3
"""

from .base_storage import BaseProcessorStorage, RawPartition
from .local_storage import LocalProcessorStorage
from .minio_storage import MinioProcessorStorage

__all__ = [
    "BaseProcessorStorage",
    "RawPartition",
    "LocalProcessorStorage",
    "MinioProcessorStorage",
]
