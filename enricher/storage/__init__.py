from .base_storage import BaseEnricherStorage, CleanPartition
from .local_storage import LocalEnricherStorage

# MinioEnricherStorage : import direct depuis enricher.storage.minio_storage
# (import lazy pour eviter minio_config.MINIO_CONFIG si .env absent)

__all__ = [
    "BaseEnricherStorage",
    "CleanPartition",
    "LocalEnricherStorage",
]
