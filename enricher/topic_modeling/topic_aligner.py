"""
TopicAligner : aligne les IDs de topics entre deux fits BERTopic consecutifs.

Algorithme :
  1. Calcule la similarite cosinus entre les centroides du nouveau modele
     et ceux de l'ancien modele (ancre dans model_store.centroids).
  2. Appariement greedy : chaque nouveau topic est mappe sur l'ancien
     le plus similaire (non encore apparie), si la similarite depasse le seuil.
  3. Les nouveaux topics sans correspondance (topics emergents) conservent
     leur ID natif BERTopic.
"""

from __future__ import annotations

import numpy as np


class TopicAligner:
    """
    Aligne les topic_ids entre deux fits BERTopic successifs.
    """

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        self._threshold = similarity_threshold

    def align(
        self,
        old_centroids: np.ndarray,
        new_centroids: np.ndarray,
    ) -> dict[int, int]:
        """
        Cree le mapping new_topic_id : old_topic_id par similarite cosinus.

        Args:
            old_centroids : Centroides de l'ancien modele, shape (n_old, dim).
            new_centroids : Centroides du nouveau modele, shape (n_new, dim).
        """
        if old_centroids.size == 0 or new_centroids.size == 0:
            return {}

        sim = self._cosine_similarity(new_centroids, old_centroids)
        # sim[i, j] = similarite entre new_centroids[i] et old_centroids[j]

        mapping: dict[int, int] = {}
        used_old: set[int] = set()

        # Greedy : par ordre decroissant de similarite max
        scores = [
            (sim[i].max(), i, int(sim[i].argmax())) for i in range(len(new_centroids))
        ]
        scores.sort(reverse=True)

        for best_sim, new_id, old_id in scores:
            if best_sim < self._threshold:
                break

            if old_id in used_old:
                # Chercher le meilleur candidat non encore apparié
                old_id = next(
                    (
                        int(cid)
                        for cid in np.argsort(sim[new_id])[::-1]
                        if sim[new_id, cid] >= self._threshold and cid not in used_old
                    ),
                    None,
                )
                if old_id is None:
                    continue

            mapping[new_id] = old_id
            used_old.add(old_id)

        return mapping

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrice de similarite cosinus (n_a, n_b)."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return a_norm @ b_norm.T
