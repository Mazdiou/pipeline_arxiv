"""
TopicProbabilities : calcule les probabilités d'appartenance au topic final pour chaque document,
en gérant les différents formats de sortie de BERTopic et les outliers réassignés par reduce_outliers.
"""

from typing import Any

import numpy as np

from .topic_result import TopicResult


def build_topic_probs(
    probs_raw: Any,
    topics_raw: list[int],
    topics_final: list[int],
    outlier_sims: list[float],
) -> list[float]:
    """
    Construit les probabilités d'appartenance au topic final.
    - Si probs_raw est None, on considère que les documents non-outliers ont une probabilité de 1.0, et les outliers 0.0.
    - Si probs_raw est un tableau 1D, on l'utilise directement (en le clipant entre 0 et 1).
    - Si probs_raw est un tableau 2D, on prend la probabilité du topic assigné pour chaque document (en gérant les outliers).
    """
    if probs_raw is None:
        base = [1.0 if t != -1 else 0.0 for t in topics_raw]
    else:
        arr = np.asarray(probs_raw)
        if arr.ndim == 1:
            base = [float(np.clip(p, 0.0, 1.0)) for p in arr]
        else:
            base = [
                (
                    float(np.clip(arr[i, t], 0.0, 1.0))
                    if t != -1 and t < arr.shape[1]
                    else 0.0
                )
                for i, t in enumerate(topics_raw)
            ]

    return [
        0.0 if final == -1 else (sim if raw == -1 else p)
        for raw, final, p, sim in zip(topics_raw, topics_final, base, outlier_sims)
    ]


def all_outliers(n: int) -> "TopicResult":
    """
    Retourne un TopicResult où tous les documents sont outliers.
    """

    labels = ["misc"] * n

    return TopicResult(
        topic_ids=[-1] * n,
        topic_labels=labels,
        topic_probs=[0.0] * n,
    )
