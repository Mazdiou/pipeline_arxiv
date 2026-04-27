"""
Réduction des outliers dans les résultats de topic modeling BERTopic.
"""

from __future__ import annotations
import numpy as np


def reduce_outliers(
    model, topics: list[int], embeddings: np.ndarray, threshold: float
) -> tuple[list[int], list[float]]:
    """
    Réassigne les documents topic=-1 au topic le plus proche par similarité cosinus.
    Retourne (new_topics, sims) où sims[i] = -1.0 si non-outlier, similarité si réassigné.
    """
    n = len(topics)
    sims: list[float] = [-1.0] * n  # sentinel : non-outlier

    outlier_indices = [i for i, t in enumerate(topics) if t == -1]
    if not outlier_indices:
        return topics, sims

    for i in outlier_indices:
        sims[i] = 0.0  # non réassigné par défaut

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Source primaire : centroïdes calculés depuis le batch courant
    # Fallback : topic_embeddings_ du modèle BERTopic
    centroid_map = _build_centroid_map(topics, embeddings)
    if centroid_map:
        topic_ids, vectors = zip(*sorted(centroid_map.items()))
        topic_matrix = np.vstack(vectors)
    else:
        topic_matrix, topic_ids = _topic_embeddings_from_model(model)

    # Similarité cosinus vectorisée
    outlier_embs = embeddings[outlier_indices]
    outlier_embs_norm = outlier_embs / (
        np.linalg.norm(outlier_embs, axis=1, keepdims=True) + 1e-12
    )
    topic_matrix_norm = topic_matrix / (
        np.linalg.norm(topic_matrix, axis=1, keepdims=True) + 1e-12
    )
    sim = np.dot(outlier_embs_norm, topic_matrix_norm.T)

    # Réassignation si similarité >= threshold
    new_topics = list(topics)
    best_idx = np.argmax(sim, axis=1)
    best_sim = sim[np.arange(len(outlier_indices)), best_idx]

    for idx, sim_val, topic_idx in zip(outlier_indices, best_sim, best_idx):
        if sim_val >= threshold:
            new_topics[idx] = topic_ids[topic_idx]
            sims[idx] = float(np.clip(sim_val, 0.0, 1.0))

    return new_topics, sims


def _build_centroid_map(
    topics: list[int], embeddings: np.ndarray
) -> dict[int, np.ndarray]:
    """Calcule le centroïde moyen par topic depuis le batch courant."""
    topics_arr = np.asarray(topics)
    return {
        tid: embeddings[topics_arr == tid].mean(axis=0)
        for tid in set(topics)
        if tid != -1
    }


def _topic_embeddings_from_model(model) -> tuple[np.ndarray, list[int]]:
    """
    Extrait les embeddings des topics valides depuis un modèle BERTopic.
    Utilise get_topic_info() pour identifier les IDs valides (robuste aux versions).
    """
    topic_embs = getattr(model, "topic_embeddings_", None)
    if topic_embs is None or len(topic_embs) == 0:
        raise ValueError("model.topic_embeddings_ absent ou vide.")

    topic_info = model.get_topic_info()
    valid_ids = sorted(topic_info.loc[topic_info["Topic"] != -1, "Topic"].tolist())
    if not valid_ids:
        raise ValueError("Aucun topic valide dans le modèle.")

    # Indexation explicite par topic_id pour éviter les décalages liés à _outliers
    offset = getattr(model, "_outliers", 0)
    matrix = np.asarray(
        [topic_embs[tid + offset] for tid in valid_ids], dtype=np.float32
    )

    return matrix, valid_ids
