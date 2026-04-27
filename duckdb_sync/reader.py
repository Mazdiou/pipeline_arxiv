"""
Abstraction de lecture des données GOLD + TRENDS.

Deux implémentations disponibles :
    - LocalReader  : lit les fichiers Parquet directement depuis le filesystem local
    - MinioReader  : télécharge les objets depuis MinIO en mémoire (BytesIO)

Les loaders (articles.py, topics.py, trends.py) reçoivent des DataFrames
depuis le reader — ils ne connaissent pas la provenance des données.

Structure des données attendue :
GOLD   : {gold_bucket ou gold_path}/{category}/{YYYY}/{MM}/articles.parquet
TRENDS : {trends_bucket ou trends_path}/{category}/history.parquet
         {trends_bucket ou trends_path}/{category}/reports/*.parquet
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class BaseDataReader(ABC):
    """Interface de lecture des données GOLD et TRENDS."""

    @abstractmethod
    def discover_categories(self) -> list[str]:
        """
        Retourne la liste des catégories disponibles dans GOLD.

        Returns
        -------
        list[str]
            Noms des catégories (ex: ["cs_AI", "stat_ML"]) triés par ordre alphabétique.
        """

    @abstractmethod
    def read_gold_partitions(self, category: str) -> list[pd.DataFrame]:
        """
        Retourne tous les DataFrames GOLD d'une catégorie.

        Chaque DataFrame correspond à une partition temporelle (YYYY/MM).

        Parameters
        ----------
        category : str
            Nom de la catégorie (ex: "cs_AI").

        Returns
        -------
        list[pd.DataFrame]
            Liste des DataFrames lus (peut être vide si aucun fichier trouvé).
        """

    @abstractmethod
    def read_history(self, category: str) -> pd.DataFrame:
        """
        Retourne le DataFrame history.parquet d'une catégorie.

        Parameters
        ----------
        category : str
            Nom de la catégorie.

        Returns
        -------
        pd.DataFrame
            DataFrame de l'historique, vide si le fichier est absent ou illisible.
        """

    @abstractmethod
    def read_reports(self, category: str) -> list[pd.DataFrame]:
        """
        Retourne tous les DataFrames de rapports de tendance d'une catégorie.

        Chaque DataFrame correspond à un fichier reports/*.parquet.

        Parameters
        ----------
        category : str
            Nom de la catégorie.

        Returns
        -------
        list[pd.DataFrame]
            Liste des DataFrames des rapports (peut être vide).
        """


# Implémentation locale (filesystem)
class LocalReader(BaseDataReader):
    """
    Lit les données depuis le filesystem local.

    Parameters
    ----------
    gold_path : str
        Chemin racine de la zone GOLD (ex: "/data/gold").
    trends_path : str
        Chemin racine de la zone TRENDS (ex: "/data/trends").
    """

    def __init__(self, gold_path: str, trends_path: str) -> None:
        self._gold_path = Path(gold_path)
        self._trends_path = Path(trends_path)

    # Implémentation de BaseDataReader
    def discover_categories(self) -> list[str]:
        if not self._gold_path.exists():
            return []
        # Retourne uniquement les sous-dossiers (catégories) triés
        return sorted(d.name for d in self._gold_path.iterdir() if d.is_dir())

    def read_gold_partitions(self, category: str) -> list[pd.DataFrame]:
        category_dir = self._gold_path / category
        if not category_dir.exists():
            logger.warning("Dossier GOLD absent pour '%s'.", category)
            return []

        frames: list[pd.DataFrame] = []
        # Cherche tous les fichiers articles.parquet récursivement
        for parquet_file in sorted(category_dir.rglob("articles.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
                if not df.empty:
                    frames.append(df)
                    logger.debug(
                        "  GOLD lu : %s (%d articles)", parquet_file.parent, len(df)
                    )
            except Exception as exc:
                logger.warning("Lecture GOLD échouée '%s' : %s", parquet_file, exc)

        return frames

    def read_history(self, category: str) -> pd.DataFrame:
        history_path = self._trends_path / category / "history.parquet"
        if not history_path.exists():
            logger.warning("history.parquet absent pour '%s'.", category)
            return pd.DataFrame()
        try:
            return pd.read_parquet(history_path)
        except Exception as exc:
            logger.warning(
                "Lecture history.parquet échouée pour '%s' : %s", category, exc
            )
            return pd.DataFrame()

    def read_reports(self, category: str) -> list[pd.DataFrame]:
        reports_dir = self._trends_path / category / "reports"
        if not reports_dir.exists():
            return []

        frames: list[pd.DataFrame] = []
        for report_file in sorted(reports_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(report_file)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.warning("Lecture rapport échouée '%s' : %s", report_file, exc)

        return frames


# Implémentation MinIO
class MinioReader(BaseDataReader):
    """
    Lit les données depuis MinIO en téléchargeant les objets en mémoire.

    Utilise le même pattern lazy que MinioEnricherStorage :
    le client MinIO est initialisé au premier appel I/O.

    Parameters
    ----------
    gold_bucket : str
        Bucket MinIO de la zone GOLD (ex: 'arxiv-gold').
    trends_bucket : str
        Bucket MinIO de la zone TRENDS (ex: 'arxiv-trends').
    endpoint : str
        Adresse du serveur MinIO (ex: 'localhost:9000').
    access_key : str
        Clé d'accès MinIO.
    secret_key : str
        Clé secrète MinIO.
    secure : bool, default=False
        True → utilise HTTPS, False → HTTP.
    """

    def __init__(
        self,
        gold_bucket: str,
        trends_bucket: str,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ) -> None:
        self._gold_bucket = gold_bucket
        self._trends_bucket = trends_bucket
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure
        self._client = None  # client MinIO initialisé à la demande (lazy)

    @classmethod
    def from_minio_config(cls) -> "MinioReader":
        """
        Construit un MinioReader depuis la configuration MINIO_CONFIG (.env).

        Returns
        -------
        MinioReader
            Instance configurée avec les paramètres du fichier minio_config.py.
        """
        from minio_config import MINIO_CONFIG

        return cls(
            gold_bucket=MINIO_CONFIG.gold_bucket,
            trends_bucket=MINIO_CONFIG.trends_bucket,
            endpoint=MINIO_CONFIG.endpoint,
            access_key=MINIO_CONFIG.access_key,
            secret_key=MINIO_CONFIG.secret_key,
            secure=MINIO_CONFIG.secure,
        )

    # Implémentation de BaseDataReader
    def discover_categories(self) -> list[str]:
        client = self._ensure_client()
        categories: set[str] = set()
        try:
            # Liste les objets à la racine du bucket (non récursif)
            for obj in client.list_objects(self._gold_bucket, recursive=False):
                name = obj.object_name.rstrip("/")
                if name:
                    categories.add(name.split("/")[0])
        except Exception as exc:
            logger.error("Listing MinIO '%s' échoué : %s", self._gold_bucket, exc)
        return sorted(categories)

    def read_gold_partitions(self, category: str) -> list[pd.DataFrame]:
        client = self._ensure_client()
        frames: list[pd.DataFrame] = []
        prefix = f"{category}/"

        try:
            objects = client.list_objects(
                self._gold_bucket, prefix=prefix, recursive=True
            )
            for obj in objects:
                key = obj.object_name
                if not key.endswith("articles.parquet"):
                    continue
                df = self._download_parquet(self._gold_bucket, key)
                if df is not None and not df.empty:
                    frames.append(df)
                    logger.debug("  MinIO GOLD lu : %s (%d articles)", key, len(df))
        except Exception as exc:
            logger.error("Listing GOLD MinIO pour '%s' échoué : %s", category, exc)

        return frames

    def read_history(self, category: str) -> pd.DataFrame:
        key = f"{category}/history.parquet"
        df = self._download_parquet(self._trends_bucket, key)
        if df is None:
            logger.warning("history.parquet absent dans MinIO pour '%s'.", category)
            return pd.DataFrame()
        return df

    def read_reports(self, category: str) -> list[pd.DataFrame]:
        client = self._ensure_client()
        prefix = f"{category}/reports/"
        frames: list[pd.DataFrame] = []

        try:
            objects = client.list_objects(
                self._trends_bucket, prefix=prefix, recursive=True
            )
            # Tri par nom d'objet pour avoir un ordre déterministe
            for obj in sorted(objects, key=lambda o: o.object_name):
                if not obj.object_name.endswith(".parquet"):
                    continue
                df = self._download_parquet(self._trends_bucket, obj.object_name)
                if df is not None and not df.empty:
                    frames.append(df)
        except Exception as exc:
            logger.error("Listing reports MinIO pour '%s' échoué : %s", category, exc)

        return frames

    # Méthodes internes
    def _download_parquet(self, bucket: str, key: str) -> pd.DataFrame | None:
        """
        Télécharge un objet Parquet depuis MinIO et le charge en DataFrame.

        Parameters
        ----------
        bucket : str
            Nom du bucket MinIO.
        key : str
            Chemin de l'objet dans le bucket.

        Returns
        -------
        pd.DataFrame | None
            DataFrame si la lecture réussit, None en cas d'erreur.
        """
        client = self._ensure_client()
        response = None
        try:
            response = client.get_object(bucket, key)
            data = response.read()
            return pd.read_parquet(io.BytesIO(data))
        except Exception as exc:
            logger.warning("Téléchargement MinIO '%s/%s' échoué : %s", bucket, key, exc)
            return None
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def _ensure_client(self):
        """
        Initialise et retourne le client MinIO (lazy).

        Returns
        -------
        Minio
            Client MinIO configuré.
        """
        if self._client is not None:
            return self._client
        try:
            from minio import Minio
        except ImportError as exc:
            raise ImportError(
                "minio est requis pour le backend MinIO. "
                "Installation : pip install minio"
            ) from exc

        self._client = Minio(
            self._endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
        )
        return self._client
