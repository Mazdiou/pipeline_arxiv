"""
Métriques de dérive sémantique pour détecter les changements de sens d'un topic dans le temps.

Notre approche DST :
    Pour chaque topic k et chaque mois t :
        centroid(k, t) = mean(embeddings des articles du topic k publiés au mois t)
    On suit l'évolution de ce centroïde dans l'espace d'embeddings.

Métriques implémentées :
    1. semantic_drift  : dérive cosinus entre centroids consécutifs
    2. keyword_drift   : dérive lexicale par similarité de Jaccard
    3. stability_score : score global e^(-mean_drift)
"""

from __future__ import annotations

import math

import numpy as np


# 1. Dérive sémantique (DST)
def compute_semantic_drift(
    centroids_by_month: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Dérive sémantique : mesure si un topic change de sens d'un mois à l'autre.

    Formule :
        drift(t) = 1 - cosine_similarity(centroid_t, centroid_{t-1})

    Le centroïde d'un topic au mois t = moyenne des embeddings
    de tous ses articles publiés ce mois-là.

    Interprétation :
        drift ≈ 0.0  : le topic traite du même sujet (sémantiquement stable)
        drift ≈ 0.2  : légère évolution des concepts abordés
        drift ≈ 0.5  : transformation notable du sujet
        drift ≈ 1.0  : le topic a complètement changé de sujet (TRANSFORMING)

    Exemple concret :
        Un topic "machine learning" qui évolue progressivement vers "LLM fine-tuning"
        entre 2020 et 2023 montrera un drift faible mais cumulatif sur cette période.

    Args:
        centroids_by_month : dict["YYYY-MM"] : centroïde.
                             Les clés doivent être triables chronologiquement.

    Returns:
        dict["YYYY-MM"] : drift ∈ [0, 1].
        Le premier mois est absent du résultat (pas de mois précédent à comparer).
    """
    months = sorted(centroids_by_month.keys())
    result: dict[str, float] = {}

    for i in range(1, len(months)):
        prev_month = months[i - 1]
        curr_month = months[i]
        cosine_sim = _cosine_similarity(
            centroids_by_month[prev_month],
            centroids_by_month[curr_month],
        )
        result[curr_month] = round(max(0.0, 1.0 - cosine_sim), 4)

    return result


# 2. Dérive lexicale
def compute_keyword_drift(
    keywords_by_month: dict[str, list[str]],
) -> dict[str, float]:
    """
    Dérive lexicale : mesure si le vocabulaire dominant d'un topic évolue.

    Formule :
        keyword_drift(t) = 1 - Jaccard(tokens_t, tokens_{t-1})

    Args:
        keywords_by_month : dict["YYYY-MM"] : liste des mots-clés dominants du topic.

    Returns:
        dict["YYYY-MM"] : drift ∈ [0, 1].
        Le premier mois est absent du résultat (pas de mois précédent).
    """
    months = sorted(keywords_by_month.keys())
    result: dict[str, float] = {}

    for i in range(1, len(months)):
        prev_month = months[i - 1]
        curr_month = months[i]
        tokens_prev = _tokenize_keywords(keywords_by_month[prev_month])
        tokens_curr = _tokenize_keywords(keywords_by_month[curr_month])
        jaccard = _jaccard_similarity(tokens_prev, tokens_curr)
        result[curr_month] = round(1.0 - jaccard, 4)

    return result


# 3. Score de stabilité
def compute_stability_score(drift_values: list[float]) -> float:
    """
    Score de stabilité globale d'un topic : résumé exponentiel de la dérive.

    Formule :
        stability = e^(-mean_semantic_drift)

    Les petites dérives sont peu pénalisées, les grandes sont fortement pénalisées.
    La formule linéaire aurait une pente constante, moins discriminante.

    Args:
        drift_values : Liste des valeurs de dérive sémantique par période.

    Returns:
        Score ∈ (0, 1]. Retourne 1.0 (stable par défaut) si liste vide.
    """
    if not drift_values:
        return 1.0

    mean_drift = float(np.mean(drift_values))
    return round(math.exp(-mean_drift), 4)


# Helpers
def _tokenize_keywords(keywords: list[str]) -> set[str]:
    """
    Tokenise une liste de mots-clés en tokens individuels (minuscules).

    Découpe les expressions multi-mots en tokens pour un Jaccard plus
    discriminant : "federated learning framework" : {"federated", "learning", "framework"}.
    Les stop-words courts (≤ 2 chars) sont filtrés pour réduire le bruit.
    """
    tokens: set[str] = set()
    for kw in keywords:
        for token in kw.lower().split():
            if len(token) > 2:
                tokens.add(token)
    return tokens


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similarité cosinus entre deux vecteurs.

    cos(θ) = (a · b) / (‖a‖ · ‖b‖)

    Retourne 0.0 si l'un des vecteurs est nul (division par zéro impossible).
    Clampé dans [-1, 1] pour éviter les erreurs numériques.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Similarité de Jaccard entre deux ensembles.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Convention : deux ensembles vides ont une similarité de 1.0 (identiques).
    Un ensemble vide face à un non-vide a une similarité de 0.0.
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
