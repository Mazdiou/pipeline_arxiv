"""
Backend de stockage MinIO (compatible S3) pour la zone RAW.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from minio import Minio
from minio.error import S3Error

from minio_config import MINIO_CONFIG
from .base_storage import BaseStorage, _group_by_month, _merge, _safe_category_name


# Storage MinIO
class MinioStorage(BaseStorage):
    """
    Stockage RAW sur MinIO (compatible S3).

    Structure des objets :

        {bucket}/
        └── {safe_category}/
            └── {YYYY}/
                └── {MM}/
                    └── articles.json
    """

    def __init__(
        self,
        endpoint: str = MINIO_CONFIG.endpoint,
        access_key: str = MINIO_CONFIG.access_key,
        secret_key: str = MINIO_CONFIG.secret_key,
        bucket: str = MINIO_CONFIG.raw_bucket,
        secure: bool = MINIO_CONFIG.secure,
        overwrite: bool = MINIO_CONFIG.overwrite,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        self.bucket = bucket
        self.overwrite = overwrite
        self.logger = logger or logging.getLogger("arxiv.collector.storage")

        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        self._ensure_bucket()

    # Interface BaseStorage
    def save(self, articles: list[dict], category: str, run_date: datetime) -> str:
        """
        Sauvegarde les articles groupés par mois de publication.

        Returns:
            Chemin objet complet :
            "{bucket}/{category}/{YYYY}/{MM}/articles.json"
        """

        articles_by_month = _group_by_month(articles, run_date, self.logger)

        last_key: Optional[str] = None
        fallback_key = self._build_key(category, run_date)

        for month_dt, month_articles in articles_by_month.items():

            key = self._build_key(category, month_dt)

            if self.overwrite or not self._object_exists(key):
                final_articles = month_articles
                new_count = len(month_articles)
                duplicate_count = 0
                updated_count = 0

            else:

                existing_articles = self._download(key)

                final_articles, new_count, duplicate_count, updated_count = _merge(
                    existing_articles,
                    month_articles,
                    self.logger,
                )

            month_str = month_dt.strftime("%Y-%m")

            self.logger.info(
                f"[{month_str}] Stockage MinIO : "
                f"{new_count} nouveaux, "
                f"{updated_count} versions mises à jour, "
                f"{duplicate_count} doublons ignorés, "
                f"{self.bucket}/{key}"
            )

            self._upload(key, final_articles, new_count, updated_count)

            last_key = key

        return f"{self.bucket}/{last_key or fallback_key}"

    def load(self, path: str) -> list[dict]:
        """
        Charge les articles depuis un objet MinIO.

        Args:
            path : "{bucket}/{category}/{YYYY}/{MM}/articles.json"
        """

        normalized = path.replace("\\", "/")
        parts = normalized.split("/")

        key = "/".join(parts[1:])  # suppression bucket

        return self._download(key)

    # Gestion bucket
    def _ensure_bucket(self) -> None:
        """Crée le bucket s'il n'existe pas."""

        try:
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)
                self.logger.info(f"Bucket MinIO créé : '{self.bucket}'")
            else:
                self.logger.debug(f"Bucket MinIO existant : '{self.bucket}'")

        except S3Error as exc:
            self.logger.error(
                f"Impossible de vérifier/créer le bucket '{self.bucket}': {exc}"
            )
            raise

    # Construction des clés objets
    def _build_key(self, category: str, ref_date: datetime) -> str:
        """
        Construit la clé objet MinIO.

        Format :
            {safe_category}/{YYYY}/{MM}/articles.json
        """

        safe_category = _safe_category_name(category)
        date_path = ref_date.strftime("%Y/%m")

        return f"{safe_category}/{date_path}/articles.json"

    # Opérations MinIO
    def _object_exists(self, key: str) -> bool:
        """
        Vérifie l'existence d'un objet dans le bucket.
        """

        try:
            self._client.stat_object(self.bucket, key)
            return True

        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise

    def _download(self, key: str) -> list[dict]:
        """
        Télécharge et parse un objet JSON.

        Retourne [] si l'objet est absent ou illisible.
        """

        response = None
        try:
            response = self._client.get_object(self.bucket, key)

            raw_data = response.read().decode("utf-8")
            data = json.loads(raw_data)
            articles = data.get("articles", [])

            self.logger.debug(
                f"Objet MinIO chargé : {len(articles)} articles "
                f"({self.bucket}/{key})"
            )

            return articles

        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return []

            self.logger.error(
                f"S3Error lors du téléchargement "
                f"{self.bucket}/{key} (code={exc.code}): {exc} — "
                f"merge impossible, l'objet sera écrasé"
            )

            return []

        except (json.JSONDecodeError, KeyError) as exc:
            self.logger.warning(
                f"Objet MinIO illisible ({exc}) → écrasement " f"{self.bucket}/{key}"
            )

            return []

        finally:
            if response:
                response.close()
                response.release_conn()

    def _upload(
        self,
        key: str,
        articles: list[dict],
        new_count: int,
        updated_count: int = 0,
    ) -> None:
        """
        Upload d'un objet JSON dans MinIO.
        """

        payload = {
            "_metadata": {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "total_articles": len(articles),
                "new_articles_this_run": new_count,
                "updated_articles_this_run": updated_count,
                "object_key": f"{self.bucket}/{key}",
            },
            "articles": articles,
        }

        json_bytes = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        try:
            self._client.put_object(
                self.bucket,
                key,
                io.BytesIO(json_bytes),
                length=len(json_bytes),
                content_type="application/json",
            )

            self.logger.info(
                f"Objet MinIO sauvegardé → "
                f"{self.bucket}/{key} "
                f"({len(articles)} articles au total)"
            )

        except S3Error as exc:
            self.logger.error(f"Erreur upload MinIO ({key}): {exc}")
            raise
