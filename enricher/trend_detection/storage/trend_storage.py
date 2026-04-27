"""
Contrat abstrait + implémentations de la persistance TRENDS.

Deux couches de persistance par catégorie :

  1. Snapshots bruts      : {category}/snapshots/{YYYY-MM}.parquet
     Agrégation mensuelle brute (vérité immuable). Source pour rebuild_history().

  2. History Prophet-ready : {category}/history.parquet
     Séries temporelles cumulatives (topic_id, month, métriques).
     Reconstruit depuis snapshots/ après un re-fit BERTopic.

Deux implémentations :
  LocalTrendStorage : système de fichiers local
  MinioTrendStorage : bucket MinIO (arxiv-trends, pattern MINIO_CONFIG)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# Contrat abstrait
class BaseTrendStorage(ABC):
    """Interface abstraite pour la persistance des snapshots et de l'historique."""

    @abstractmethod
    def write_snapshot(
        self, snapshot_df: pd.DataFrame, category: str, month: str
    ) -> None:
        """Ecrit le snapshot mensuel brut (6 champs)."""

    @abstractmethod
    def read_snapshot(self, category: str, month: str) -> pd.DataFrame:
        """Lit le snapshot d'un mois. Retourne df vide si absent."""

    @abstractmethod
    def snapshot_exists(self, category: str, month: str) -> bool:
        """Retourne True si le snapshot existe."""

    @abstractmethod
    def read_history(self, category: str) -> pd.DataFrame:
        """Lit le history cumulatif. Retourne df vide si absent."""

    @abstractmethod
    def update_history(
        self, snapshot_df: pd.DataFrame, category: str, month: str
    ) -> None:
        """Upsert du snapshot mensuel dans history (remplace le mois si existant)."""

    @abstractmethod
    def rebuild_history(self, category: str) -> None:
        """Reconstruit history depuis tous les snapshots (apres re-fit BERTopic)."""

    @abstractmethod
    def write_report(self, report_df: pd.DataFrame, category: str, month: str) -> None:
        """Persiste le rapport mensuel (une ligne par topic, séries exclues)."""

    def resolve_labels(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """
        Remplace les labels génériques (topic_N) par le dernier label descriptif connu.

        Met à jour le registre avec tout nouveau label descriptif rencontré.
        Opère en place sur la colonne topic_label.
        """
        registry = self._load_label_registry(category)
        updated = False

        for i, row in df.iterrows():
            tid = int(row["topic_id"])
            label = row["topic_label"]
            is_generic = bool(__import__("re").match(r"^topic_\d+$", str(label)))

            if not is_generic:
                # Label descriptif : mettre à jour le registre
                if registry.get(tid) != label:
                    registry[tid] = label
                    updated = True
            else:
                # Label générique : utiliser le registre si disponible
                if tid in registry:
                    df.at[i, "topic_label"] = registry[tid]

        if updated:
            self._save_label_registry(category, registry)

        return df

    @abstractmethod
    def _load_label_registry(self, category: str) -> dict:
        """Charge le registre topic_id : label canonique. Retourne {} si absent."""

    @abstractmethod
    def _save_label_registry(self, category: str, registry: dict) -> None:
        """Persiste le registre topic_id : label canonique."""


# Implementation locale
class LocalTrendStorage(BaseTrendStorage):
    """
    Persistance locale des trends sur le systeme de fichiers.

    Args:
        trends_path : Dossier racine TRENDS (ex: 'data/TRENDS').
    """

    _SNAPSHOTS_DIR = "snapshots"
    _HISTORY_FILE = "history.parquet"

    def __init__(self, trends_path: str) -> None:
        self._root = Path(trends_path)

    # Snapshots
    def write_snapshot(
        self, snapshot_df: pd.DataFrame, category: str, month: str
    ) -> None:
        path = self._snapshot_path(category, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_df.to_parquet(path, index=False, compression="snappy")
        logger.debug("Snapshot ecrit : %s", path)

    def read_snapshot(self, category: str, month: str) -> pd.DataFrame:
        path = self._snapshot_path(category, month)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def snapshot_exists(self, category: str, month: str) -> bool:
        return self._snapshot_path(category, month).exists()

    # History
    def read_history(self, category: str) -> pd.DataFrame:
        path = self._history_path(category)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def update_history(
        self, snapshot_df: pd.DataFrame, category: str, month: str
    ) -> None:
        path = self._history_path(category)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            existing = existing[existing["month"] != month]
            combined = pd.concat([existing, snapshot_df], ignore_index=True)
        else:
            combined = snapshot_df

        combined = combined.sort_values(["topic_id", "month"], ignore_index=True)
        combined.to_parquet(path, index=False, compression="snappy")
        logger.info(
            "History mis à jour : %s/%s (%d topics, %d entrées totales)",
            category,
            month,
            len(snapshot_df),
            len(combined),
        )

    def rebuild_history(self, category: str) -> None:
        snapshots_dir = self._root / category / self._SNAPSHOTS_DIR
        if not snapshots_dir.exists():
            logger.warning(
                "Pas de snapshots pour '%s' : history non reconstruit.", category
            )
            return

        parquet_files = sorted(snapshots_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning("Repertoire snapshots vide pour '%s'.", category)
            history_path = self._history_path(category)
            if history_path.exists():
                history_path.unlink()
            return

        frames = []
        for f in parquet_files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as exc:
                logger.warning("Snapshot illisible '%s' : %s", f, exc)

        if not frames:
            return

        history = pd.concat(frames, ignore_index=True)
        history = history.sort_values(["topic_id", "month"], ignore_index=True)

        path = self._history_path(category)
        path.parent.mkdir(parents=True, exist_ok=True)
        history.to_parquet(path, index=False, compression="snappy")
        logger.info(
            "History reconstruit pour '%s' : %d entrées depuis %d snapshots.",
            category,
            len(history),
            len(parquet_files),
        )

    # Helpers
    def write_report(self, report_df: pd.DataFrame, category: str, month: str) -> None:
        path = self._root / category / "reports" / f"{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_parquet(path, index=False, compression="snappy")
        logger.debug("Rapport écrit : %s", path)

    def _load_label_registry(self, category: str) -> dict:
        import json

        path = self._root / category / "label_registry.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    def _save_label_registry(self, category: str, registry: dict) -> None:
        import json

        path = self._root / category / "label_registry.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in registry.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.debug("Registre labels mis à jour : %s", path)

    def _snapshot_path(self, category: str, month: str) -> Path:
        return self._root / category / self._SNAPSHOTS_DIR / f"{month}.parquet"

    def _history_path(self, category: str) -> Path:
        return self._root / category / self._HISTORY_FILE


# Implémentation MinIO
class MinioTrendStorage(BaseTrendStorage):
    """
    Persistance des trends dans un bucket MinIO (arxiv-trends).

    Structure des clés :
      {category}/snapshots/{YYYY-MM}.parquet   ← snapshots bruts mensuels
      {category}/history.parquet               ← historique cumulatif

    Les paramètres de connexion proviennent de MINIO_CONFIG (.env) par défaut.
    Client MinIO lazy-initialisé au premier appel I/O.
    """

    _SNAPSHOTS_PREFIX = "snapshots"
    _HISTORY_KEY = "history.parquet"

    def __init__(
        self,
        bucket: str | None = None,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool | None = None,
    ) -> None:
        from minio_config import MINIO_CONFIG

        self._bucket = bucket or MINIO_CONFIG.trends_bucket
        self._endpoint = endpoint or MINIO_CONFIG.endpoint
        self._access_key = access_key or MINIO_CONFIG.access_key
        self._secret_key = secret_key or MINIO_CONFIG.secret_key
        self._secure = secure if secure is not None else MINIO_CONFIG.secure
        self._client: Any = None  # lazy

    # Snapshots
    def write_snapshot(
        self, snapshot_df: pd.DataFrame, category: str, month: str
    ) -> None:
        key = self._snapshot_key(category, month)
        self._put_parquet(key, snapshot_df)
        logger.debug("Snapshot MinIO écrit : %s/%s", self._bucket, key)

    def read_snapshot(self, category: str, month: str) -> pd.DataFrame:
        key = self._snapshot_key(category, month)
        return self._get_parquet(key)

    def snapshot_exists(self, category: str, month: str) -> bool:
        key = self._snapshot_key(category, month)
        try:
            self._client_().stat_object(self._bucket, key)
            return True
        except Exception:
            return False

    # History
    def read_history(self, category: str) -> pd.DataFrame:
        return self._get_parquet(self._history_key(category))

    def update_history(
        self, snapshot_df: pd.DataFrame, category: str, month: str
    ) -> None:
        existing = self.read_history(category)
        if not existing.empty:
            existing = existing[existing["month"] != month]
            combined = pd.concat([existing, snapshot_df], ignore_index=True)
        else:
            combined = snapshot_df

        combined = combined.sort_values(["topic_id", "month"], ignore_index=True)
        self._put_parquet(self._history_key(category), combined)
        logger.info(
            "History MinIO mis à jour : %s/%s (%d topics, %d entrées totales)",
            category,
            month,
            len(snapshot_df),
            len(combined),
        )

    def rebuild_history(self, category: str) -> None:
        prefix = f"{category}/{self._SNAPSHOTS_PREFIX}/"
        client = self._client_()

        try:
            objects = list(client.list_objects(self._bucket, prefix=prefix))
        except Exception as exc:
            logger.warning(
                "Listing snapshots MinIO échoué pour '%s' : %s", category, exc
            )
            return

        if not objects:
            logger.warning(
                "Pas de snapshots MinIO pour '%s' : history non reconstruit.", category
            )
            return

        frames = []
        for obj in sorted(objects, key=lambda o: o.object_name):
            df = self._get_parquet(obj.object_name)
            if not df.empty:
                frames.append(df)
            else:
                logger.warning("Snapshot MinIO illisible : %s", obj.object_name)

        if not frames:
            return

        history = pd.concat(frames, ignore_index=True)
        history = history.sort_values(["topic_id", "month"], ignore_index=True)
        self._put_parquet(self._history_key(category), history)
        logger.info(
            "History MinIO reconstruit pour '%s' : %d entrées depuis %d snapshots.",
            category,
            len(history),
            len(frames),
        )

    # Helpers privés
    def _client_(self) -> Any:
        if self._client is None:
            from minio import Minio  # noqa: PLC0415

            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
            )
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logger.info("Bucket MinIO créé : %s", self._bucket)
        return self._client

    def write_report(self, report_df: pd.DataFrame, category: str, month: str) -> None:
        key = f"{category}/reports/{month}.parquet"
        self._put_parquet(key, report_df)
        logger.debug("Rapport MinIO écrit : %s/%s", self._bucket, key)

    def _load_label_registry(self, category: str) -> dict:
        import json

        key = f"{category}/label_registry.json"
        try:
            response = self._client_().get_object(self._bucket, key)
            data = response.read()
            response.close()
            response.release_conn()
            return {int(k): v for k, v in json.loads(data).items()}
        except Exception:
            return {}

    def _save_label_registry(self, category: str, registry: dict) -> None:
        import json

        key = f"{category}/label_registry.json"
        data = json.dumps(
            {str(k): v for k, v in registry.items()}, ensure_ascii=False, indent=2
        ).encode("utf-8")
        from io import BytesIO

        self._client_().put_object(
            self._bucket,
            key,
            BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
        logger.debug("Registre labels MinIO mis à jour : %s/%s", self._bucket, key)

    def _snapshot_key(self, category: str, month: str) -> str:
        return f"{category}/{self._SNAPSHOTS_PREFIX}/{month}.parquet"

    def _history_key(self, category: str) -> str:
        return f"{category}/{self._HISTORY_KEY}"

    def _get_parquet(self, key: str) -> pd.DataFrame:
        try:
            response = self._client_().get_object(self._bucket, key)
            data = response.read()
            response.close()
            response.release_conn()
            return pd.read_parquet(BytesIO(data))
        except Exception:
            return pd.DataFrame()

    def _put_parquet(self, key: str, df: pd.DataFrame) -> None:
        buf = BytesIO()
        df.to_parquet(buf, index=False, compression="snappy")
        buf.seek(0)
        data = buf.getvalue()
        self._client_().put_object(
            self._bucket,
            key,
            BytesIO(data),
            length=len(data),
            content_type="application/octet-stream",
        )


# Factoryv
def build_trend_storage(
    storage_backend: str, trends_path: str = "data/TRENDS"
) -> BaseTrendStorage:
    """
    Instancie le TrendStorage selon storage_backend.

    Args:
        storage_backend : 'local' | 'minio' (depuis STORAGE_BACKEND dans .env).
        trends_path     : Chemin racine local (ignoré si backend='minio').

    Returns:
        LocalTrendStorage ou MinioTrendStorage.
    """
    if storage_backend == "minio":
        return MinioTrendStorage()
    return LocalTrendStorage(trends_path)


# Alias public pour backward compatibility
TrendStorage = LocalTrendStorage
