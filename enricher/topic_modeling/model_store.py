"""
TopicModelStore : gère la persistance des modèles BERTopic sur disque, un modèle par catégorie.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TopicModelStore:
    """
    Gère la persistance des modèles BERTopic sur disque, un modèle par catégorie.
    """

    _STATE_FILE = "state.json"
    _MODEL_SUBDIR = "model"

    def __init__(self, models_dir: str) -> None:
        """
        Args:
            models_dir : Dossier racine pour les modèles (créé si absent).
        """
        self._root = Path(models_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = self._load_state()

    # API publique
    def model_exists(self, category: str) -> bool:
        """
        Retourne True si un modèle BERTopic existe sur disque pour cette catégorie.
        """
        return self._model_path(category).exists()

    def load_model(self, category: str, sentence_model: Any = None) -> Any:
        """
        Charge le modèle BERTopic depuis le disque.
        """
        from bertopic import BERTopic

        path = self._model_path(category)
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle BERTopic absent pour '{category}' : {path}"
            )

        logger.debug("Chargement modèle BERTopic : %s", path)
        try:
            model = BERTopic.load(str(path), embedding_model=sentence_model)
        except Exception as exc:
            raise RuntimeError(
                f"Chargement BERTopic échoué pour '{category}' : {exc}"
            ) from exc

        logger.debug("Modèle chargé : %d topics.", len(model.get_topics()))
        return model

    def save_model(self, model: Any, category: str) -> None:
        """
        Sauvegarde le modèle BERTopic sur le disque.

        Utilise la sérialisation safetensors pour le modèle de représentation
        et pickle pour UMAP/HDBSCAN (format hybride recommandé par BERTopic).
        """
        path = self._model_path(category)
        path.mkdir(parents=True, exist_ok=True)

        logger.debug("Sauvegarde modèle BERTopic : %s", path)
        try:
            model.save(
                str(path),
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=False,  # Injecté à l'exécution, pas sauvegardé
            )
        except Exception as exc:
            # Fallback : serialisation pickle si safetensors indisponible
            logger.warning("Sauvegarde safetensors échouée (%s), fallback pickle.", exc)
            try:
                model.save(
                    str(path),
                    serialization="pickle",
                    save_ctfidf=True,
                    save_embedding_model=False,
                )
            except Exception as exc2:
                raise RuntimeError(
                    f"Sauvegarde BERTopic échouée pour '{category}' : {exc2}"
                ) from exc2

        logger.info("Modèle BERTopic sauvegardé : %s", path)

    def update_category_state(
        self,
        category: str,
        max_topic_id: int = -1,
        n_partitions: int = 0,
        last_outlier_rate: float = 0.0,
    ) -> None:
        """
        Met à jour l'état d'une catégorie et persiste state.json sur disque.
        """
        self._state[category] = {
            "max_topic_id": max_topic_id,
            "n_partitions": n_partitions,
            "last_outlier_rate": round(last_outlier_rate, 4),
            "fitted_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._save_state()
        logger.debug(
            "State mis à jour pour '%s' : max_topic_id=%d, n_partitions=%d, outlier_rate=%.2f.",
            category,
            max_topic_id,
            n_partitions,
            last_outlier_rate,
        )

    # Centroids storage
    def save_centroids(self, category: str, centroids: np.ndarray) -> None:
        """
        Sauvegarde les centroides des topics en .npy.
        """
        path = self._centroids_path(category)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), centroids)
        logger.debug("Centroides sauvegardés pour '%s' : %s.", category, path)

    def load_centroids(self, category: str) -> Optional[np.ndarray]:
        """
        Charge les centroides depuis le disque.
        """
        path = self._centroids_path(category)
        if not path.exists():
            return None
        try:
            return np.load(str(path))
        except Exception as exc:
            logger.warning(
                "Chargement centroides échoué pour '%s' : %s.", category, exc
            )
            return None

    def record_outlier_rate(self, category: str, outlier_rate: float) -> None:
        """
        Met à jour uniquement last_outlier_rate dans l'état d'une catégorie.

        Appelé après chaque transform() pour garder un historique du drift.
        Ne touche pas aux autres champs (max_topic_id, n_partitions, fitted_at).
        """
        if category not in self._state:
            self._state[category] = {}
        self._state[category]["last_outlier_rate"] = round(outlier_rate, 4)
        self._save_state()
        logger.debug(
            "Outlier rate enregistré pour '%s' : %.2f%%.",
            category,
            outlier_rate * 100,
        )

    # Helpers privés
    def _model_path(self, category: str) -> Path:
        """Chemin du dossier modèle pour une catégorie."""
        return self._root / self._safe_cat(category) / self._MODEL_SUBDIR

    def _centroids_path(self, category: str) -> Path:
        """Chemin du fichier .npy des centroides pour une catégorie."""
        return self._root / self._safe_cat(category) / "centroids.npy"

    @staticmethod
    def _safe_cat(category: str) -> str:
        """
        Convertit un nom de catégorie arXiv en nom de dossier valide.

        Exemples :
            'cs.AI'    → 'cs_AI'
        """
        return category.replace(".", "_").replace("/", "_")

    def _state_path(self) -> Path:
        return self._root / self._STATE_FILE

    def _load_state(self) -> dict[str, Any]:
        """Charge state.json depuis le disque (dict vide si absent ou corrompu)."""
        path = self._state_path()
        if not path.exists():
            return {}
        try:
            content = path.read_text(encoding="utf-8")
            return json.loads(content)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Lecture state.json échouée → reset (cause : %s).", exc)
            return {}

    def _save_state(self) -> None:
        """Persiste state.json sur disque (écriture atomique via fichier temporaire)."""
        path = self._state_path()
        tmp = path.with_suffix(".tmp.json")
        try:
            tmp.write_text(
                json.dumps(self._state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(path)  # Atomique sur la même partition
        except OSError as exc:
            logger.error("Écriture state.json échouée : %s", exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
