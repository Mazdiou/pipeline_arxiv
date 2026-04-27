"""
Implémentation MinIO du storage enricher.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from minio_config import MINIO_CONFIG

from ..pipeline.gold_schema import (
    GOLD_SCHEMA,
    GOLD_SCHEMA_COLUMNS,
    prepare_gold_dataframe,
)
from .base_storage import BaseEnricherStorage, CleanPartition

logger = logging.getLogger(__name__)

_PARQUET_SUFFIX = "/articles.parquet"


class MinioEnricherStorage(BaseEnricherStorage):
    """
    Storage MinIO pour enricher les articles arXiv.
    """

    def __init__(
        self,
        clean_bucket: str = MINIO_CONFIG.clean_bucket,
        gold_bucket: str = MINIO_CONFIG.gold_bucket,
        endpoint: str = MINIO_CONFIG.endpoint,
        access_key: str = MINIO_CONFIG.access_key,
        secret_key: str = MINIO_CONFIG.secret_key,
        secure: bool = MINIO_CONFIG.secure,
    ) -> None:

        self._clean_bucket = clean_bucket
        self._gold_bucket = gold_bucket
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure
        self._client = None

    # Implémentation de BaseEnricherStorage
    def discover(
        self,
        categories: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        months: Optional[List[int]] = None,
    ) -> List[CleanPartition]:
        """Découvre les partitions CLEAN dans le bucket MinIO arxiv-clean."""
        client = self._ensure_client()
        partitions: List[CleanPartition] = []

        try:
            objects = client.list_objects(self._clean_bucket, recursive=True)
            for obj in objects:
                key: str = obj.object_name
                if not key.endswith(_PARQUET_SUFFIX):
                    continue
                partition = _parse_key(key)
                if partition is None:
                    continue
                if categories and partition.category not in categories:
                    continue
                if years and partition.year not in years:
                    continue
                if months and partition.month not in months:
                    continue
                partitions.append(partition)
        except Exception as exc:
            logger.error(
                "Erreur listing MinIO bucket '%s' : %s", self._clean_bucket, exc
            )

        return sorted(partitions, key=lambda p: (p.category, p.year, p.month))

    def read_clean(self, partition: CleanPartition) -> pd.DataFrame:
        """Lit le Parquet CLEAN depuis MinIO."""
        client = self._ensure_client()
        key = _partition_to_key(partition)
        response = None
        try:
            response = client.get_object(self._clean_bucket, key)
            data = response.read()
        except Exception as exc:
            raise FileNotFoundError(
                f"Partition CLEAN introuvable dans MinIO : "
                f"{self._clean_bucket}/{key}"
            ) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

        return pd.read_parquet(io.BytesIO(data))

    def gold_exists(self, partition: CleanPartition) -> bool:
        """Vérifie si le fichier GOLD existe dans le bucket arxiv-gold."""
        client = self._ensure_client()
        key = _partition_to_key(partition)
        try:
            client.stat_object(self._gold_bucket, key)
            return True
        except Exception:
            return False

    def clean_is_newer_than_gold(self, partition: CleanPartition) -> bool:
        """Retourne True si l'objet CLEAN MinIO est plus récent que l'objet GOLD."""
        client = self._ensure_client()
        key = _partition_to_key(partition)
        try:
            clean_stat = client.stat_object(self._clean_bucket, key)
            gold_stat = client.stat_object(self._gold_bucket, key)
            if (
                clean_stat.last_modified is not None
                and gold_stat.last_modified is not None
            ):
                return clean_stat.last_modified > gold_stat.last_modified
            return False
        except Exception:
            return False

    def write_gold(self, df: pd.DataFrame, partition: CleanPartition) -> None:
        """Écrit le DataFrame GOLD dans le bucket arxiv-gold."""
        client = self._ensure_client()
        key = _partition_to_key(partition)

        buf, size = _dataframe_to_parquet_bytes(df)
        client.put_object(
            self._gold_bucket,
            key,
            buf,
            length=size,
            content_type="application/octet-stream",
        )
        logger.info("GOLD écrit dans MinIO : %s/%s", self._gold_bucket, key)

    # Méthodes internes
    def _ensure_client(self):
        """Initialise le client MinIO en lazy."""
        if self._client is not None:
            return self._client
        try:
            from minio import Minio
        except ImportError as exc:
            raise ImportError(
                "minio est requis pour le storage MinIO. "
                "Installation : pip install minio"
            ) from exc

        self._client = Minio(
            self._endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
        )
        self._ensure_bucket()
        return self._client

    def _ensure_bucket(self) -> None:
        """Crée le bucket arxiv-gold s'il n'existe pas."""
        try:
            if not self._client.bucket_exists(self._gold_bucket):
                self._client.make_bucket(self._gold_bucket)
                logger.info("Bucket MinIO créé : '%s'", self._gold_bucket)
        except Exception as exc:
            logger.error(
                "Impossible de vérifier/créer le bucket '%s' : %s",
                self._gold_bucket,
                exc,
            )
            raise


# Fonctions utilitaires
def _partition_to_key(partition: CleanPartition) -> str:
    """Construit la clé MinIO d'une partition : {category}/{YYYY}/{MM}/articles.parquet"""
    return (
        f"{partition.category}"
        f"/{partition.year:04d}"
        f"/{partition.month:02d}"
        f"/articles.parquet"
    )


def _parse_key(key: str) -> Optional[CleanPartition]:
    """
    Parse une clé MinIO pour en extraire la CleanPartition.

    Format attendu : {category}/{YYYY}/{MM}/articles.parquet
    Supporte les catégories composées (ex: cs/AI : category='cs_AI').
    """
    parts = key.rstrip("/").split("/")
    # Structure minimale : category/YYYY/MM/articles.parquet : 4 segments
    if len(parts) < 4:
        return None
    try:
        month = int(parts[-2])
        year = int(parts[-3])
        # Tout ce qui précède YYYY/MM/articles.parquet est la catégorie
        category = "_".join(parts[:-3])
        return CleanPartition(category=category, year=year, month=month)
    except (ValueError, IndexError):
        return None


def _dataframe_to_parquet_bytes(df: pd.DataFrame) -> tuple[io.BytesIO, int]:
    """Prépare un DataFrame GOLD et retourne (buffer, taille_en_octets)."""
    df_prepared = prepare_gold_dataframe(df)
    cols_present = [c for c in GOLD_SCHEMA_COLUMNS if c in df_prepared.columns]
    df_ordered = df_prepared[cols_present]
    schema_fields = [f for f in GOLD_SCHEMA if f.name in cols_present]
    partial_schema = pa.schema(schema_fields)

    table = pa.Table.from_pandas(
        df_ordered,
        schema=partial_schema,
        preserve_index=False,
        safe=False,
    )

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    size = buf.tell()
    buf.seek(0)
    return buf, size
