"""
Backend MinIO (compatible S3) pour le processeur arXiv.
"""

from __future__ import annotations

import io
import json
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio
from minio.error import S3Error
from minio_config import MINIO_CONFIG

from .base_storage import BaseProcessorStorage, RawPartition
from ..io.writer import CLEAN_SCHEMA, prepare_clean_dataframe


class MinioProcessorStorage(BaseProcessorStorage):
    """
    Stockage processeur sur MinIO (compatible S3).
    """

    def __init__(
        self,
        endpoint: str = MINIO_CONFIG.endpoint,
        access_key: str = MINIO_CONFIG.access_key,
        secret_key: str = MINIO_CONFIG.secret_key,
        raw_bucket: str = MINIO_CONFIG.raw_bucket,
        clean_bucket: str = MINIO_CONFIG.clean_bucket,
        secure: bool = MINIO_CONFIG.secure,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialise le stockage processeur sur MinIO.
        """
        self.raw_bucket = raw_bucket
        self.clean_bucket = clean_bucket
        self.logger = logger or logging.getLogger("arxiv.processor.storage")

        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._ensure_buckets()

    # Interface publique (BaseProcessorStorage)
    def discover(
        self,
        category: str = "",
        year: int | None = None,
        month: int | None = None,
    ) -> list[RawPartition]:
        """
        Découvre les partitions RAW dans le bucket MinIO correspondant aux filtres.
        """
        prefix = self._build_prefix(category, year, month)
        partitions: list[RawPartition] = []

        try:
            objects = self._client.list_objects(
                self.raw_bucket, prefix=prefix, recursive=True
            )
            for obj in objects:
                key = obj.object_name  # ex: "cs_AI/2026/02/articles.json"
                if not key or not key.endswith("/articles.json"):
                    continue
                parts = key.split("/")
                # Structure attendue : {category}/{YYYY}/{MM}/articles.json
                if len(parts) != 4:
                    continue
                cat, y_str, m_str, _ = parts
                try:
                    y = int(y_str)
                    m = int(m_str)
                except ValueError:
                    continue

                partitions.append(RawPartition(category=cat, year=y, month=m))

        except S3Error as exc:
            self.logger.error(
                f"Erreur listing MinIO bucket '{self.raw_bucket}' "
                f"(prefix='{prefix}'): {exc}"
            )
            raise

        return sorted(partitions, key=lambda p: (p.category, p.year, p.month))

    def read_raw(self, partition: RawPartition) -> pd.DataFrame:
        """
        Télécharge et parse le fichier articles.json RAW depuis MinIO.
        """
        key = self._raw_key(partition)
        self.logger.debug(f"Lecture RAW MinIO : {self.raw_bucket}/{key}")

        response = None
        try:
            response = self._client.get_object(self.raw_bucket, key)
            raw = json.loads(response.read().decode("utf-8"))
        except S3Error as exc:
            raise FileNotFoundError(
                f"Objet RAW introuvable dans MinIO : {self.raw_bucket}/{key} ({exc})"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON malformé dans {self.raw_bucket}/{key} : {exc}"
            ) from exc
        finally:
            if response:
                response.close()
                response.release_conn()

        articles = raw.get("articles")
        if articles is None:
            raise ValueError(
                f"Clé 'articles' absente dans {self.raw_bucket}/{key}. "
                f"Clés présentes : {list(raw.keys())}"
            )

        if not articles:
            return pd.DataFrame()

        return pd.DataFrame(articles)

    def clean_exists(self, partition: RawPartition) -> bool:
        """
        Vérifie si le fichier CLEAN Parquet existe dans le bucket MinIO.
        """
        key = self._clean_key(partition)
        try:
            self._client.stat_object(self.clean_bucket, key)
            return True
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise

    def raw_is_newer_than_clean(self, partition: RawPartition) -> bool:
        """Retourne True si l'objet RAW MinIO est plus récent que l'objet CLEAN."""
        try:
            raw_stat = self._client.stat_object(
                self.raw_bucket, self._raw_key(partition)
            )
            clean_stat = self._client.stat_object(
                self.clean_bucket, self._clean_key(partition)
            )
            if raw_stat.last_modified is None or clean_stat.last_modified is None:
                return False
            return raw_stat.last_modified > clean_stat.last_modified
        except S3Error:
            return False

    def write_clean(self, df: pd.DataFrame, partition: RawPartition) -> str:
        """
        Sérialise le DataFrame nettoyé en Parquet et l'uploade dans MinIO.
        """
        if df.empty:
            raise ValueError(
                "Impossible d'écrire un DataFrame vide en zone CLEAN MinIO."
            )

        schema_cols = [field.name for field in CLEAN_SCHEMA]
        missing = set(schema_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes pour le schéma CLEAN : {sorted(missing)}. "
            )

        # Réordonner et préparer les types (délègue à writer.prepare_clean_dataframe)
        df_ordered = prepare_clean_dataframe(df[schema_cols].copy())

        table = pa.Table.from_pandas(
            df_ordered,
            schema=CLEAN_SCHEMA,
            preserve_index=False,
            safe=False,
        )

        # Sérialisation en mémoire
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy")
        parquet_bytes = buffer.getvalue()

        key = self._clean_key(partition)
        try:
            self._client.put_object(
                self.clean_bucket,
                key,
                io.BytesIO(parquet_bytes),
                length=len(parquet_bytes),
                content_type="application/octet-stream",
            )
            self.logger.info(
                f"CLEAN MinIO sauvegardé → {self.clean_bucket}/{key} "
                f"({len(df)} articles)"
            )
        except S3Error as exc:
            self.logger.error(f"Erreur upload CLEAN MinIO ({key}): {exc}")
            raise

        return f"{self.clean_bucket}/{key}"

    # Méthodes interne
    def _ensure_buckets(self) -> None:
        """Crée les buckets RAW et CLEAN s'ils n'existent pas encore."""
        for bucket in (self.raw_bucket, self.clean_bucket):
            try:
                if not self._client.bucket_exists(bucket):
                    self._client.make_bucket(bucket)
                    self.logger.info(f"Bucket MinIO créé : '{bucket}'")
                else:
                    self.logger.debug(f"Bucket MinIO existant : '{bucket}'")
            except S3Error as exc:
                self.logger.error(
                    f"Impossible de vérifier/créer le bucket '{bucket}': {exc}"
                )
                raise

    def _raw_key(self, partition: RawPartition) -> str:
        """Clé objet RAW : {category}/{YYYY}/{MM}/articles.json"""
        return (
            f"{partition.category}/"
            f"{partition.year:04d}/{partition.month:02d}/articles.json"
        )

    def _clean_key(self, partition: RawPartition) -> str:
        """Clé objet CLEAN : {category}/{YYYY}/{MM}/articles.parquet"""
        return (
            f"{partition.category}/"
            f"{partition.year:04d}/{partition.month:02d}/articles.parquet"
        )

    def _build_prefix(
        self,
        category: str,
        year: int | None,
        month: int | None,
    ) -> str:
        """
        Construit le préfixe de listing MinIO pour optimiser la recherche.
        """
        if not category:
            return ""
        prefix = f"{category}/"
        if year is not None:
            prefix += f"{year:04d}/"
            if month is not None:
                prefix += f"{month:02d}/"
        return prefix
