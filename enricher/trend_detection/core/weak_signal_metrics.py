"""
Métriques de détection de signaux faibles sur des séries temporelles.

Un "signal faible" est une activité anormalement élevée dans un domaine
encore marginal : fort taux de croissance, faible volume absolu,
et/ou forte nouveauté sémantique des articles.

Métriques implémentées :
    1. Emergence score              — croissance forte + volume faible
    2. Novelty-weighted emergence   — émergence amplifiée par la nouveauté sémantique
    3. Persistence score            — cohérence directionnelle locale (momentum)
    4. Burst score (Kleinberg)      — détection de bursts par automate à états
"""

from __future__ import annotations

import math

import numpy as np


# 1. Emergence score
def compute_emergence_score(series: list[float]) -> list[float]:
    """
    Score d'émergence : identifie les topics à forte croissance mais faible volume.

    Formule :
        emergence_t = growth_rate_t / log(1 + n_t)

    Intuition :
        Un topic avec 5 articles qui double chaque mois est plus "émergent"
        qu'un topic avec 500 articles qui progresse de 10%.
        Le diviseur log(1 + n) pénalise les topics déjà établis (grand volume)
        pour ne conserver que les signaux naissants.

    Interprétation :
        Valeur élevée → topic naissant en accélération (candidat signal faible)
        Valeur nulle  → topic stable ou déclinant
        Valeur négative → déclin (croissance négative)

    Args:
        series : Série temporelle du nombre d'articles par mois.

    Returns:
        Score d'émergence pour chaque mois. 0.0 pour le premier mois.
    """
    n = len(series)
    if n < 2:
        return [0.0] * n

    result = [0.0]
    for i in range(1, n):
        prev = series[i - 1]
        curr = series[i]

        if prev > 0:
            growth_rate = (curr - prev) / prev
        else:
            # Pas d'articles le mois précédent → naissance du topic
            growth_rate = 1.0 if curr > 0 else 0.0

        # log(1 + n) : toujours > 0 pour n ≥ 0, croît lentement pour grands n
        volume_penalty = math.log1p(curr)
        score = growth_rate / volume_penalty if volume_penalty > 1e-10 else 0.0
        result.append(round(score, 4))

    return result


# 2. Novelty-weighted emergence
def compute_novelty_weighted_emergence(
    volume_series: list[float],
    novelty_series: list[float],
) -> list[float]:
    """
    Émergence pondérée par la nouveauté sémantique des articles.

    Formule :
        nwe_t = mean_novelty_t × growth_rate_t

    Deux dimensions complémentaires :
        - novelty_score (de l'enricher) : les articles du topic sont-ils
          sémantiquement isolés des autres clusters ?
          (1.0 = articles très "uniques", pas encore absorbés par les topics établis)
        - growth_rate : le topic croît-il rapidement en volume ?

    Un topic avec des articles sémantiquement uniques ET en forte croissance
    est un excellent candidat signal faible : il n'est pas encore "absorbé"
    par les thématiques existantes et gagne en visibilité.

    Args:
        volume_series  : n_articles par mois.
        novelty_series : mean_novelty par mois (depuis history.parquet).

    Returns:
        Score NWE pour chaque mois. 0.0 pour le premier mois.
        Retourne liste de 0.0 si les deux séries ont des longueurs différentes.
    """
    n = len(volume_series)
    if n < 2 or len(novelty_series) != n:
        return [0.0] * n

    result = [0.0]
    for i in range(1, n):
        prev = volume_series[i - 1]
        curr = volume_series[i]

        if prev > 0:
            growth_rate = (curr - prev) / prev
        else:
            growth_rate = 1.0 if curr > 0 else 0.0

        # max(0.0, ...) : la nouveauté ne peut pas être négative
        nwe = max(0.0, novelty_series[i]) * growth_rate
        result.append(round(nwe, 4))

    return result


# 3. Persistence score
def compute_persistence_score(series: list[float]) -> float:
    """
    Score de persistance directionnelle : cohérence du momentum court-terme.

    Formule :
        persistence = count(sign(Δ_t) == sign(Δ_{t-1})) / (n - 2)

    Mesure la proportion de périodes consécutives où la variation garde
    le même signe que la variation précédente.

    Différence fondamentale avec l'exposant de Hurst :
        Hurst     → mémoire LONGUE : persistance sur TOUTE la série
        Persistence → cohérence LOCALE : les hausses sont-elles immédiatement suivies
                       d'autres hausses ? (momentum à 1 pas de temps)

    Interprétation :
        1.0 → croissance parfaitement monotone (signal très fiable)
        0.5 → aussi souvent cohérent qu'incohérent (comportement aléatoire)
        0.0 → alternances parfaites hausse/baisse (bruit oscillatoire)

    Exemple :
        [10, 12, 15, 13, 16, 20] → deltas = [+2, +3, -2, +3, +4]
        Paires cohérentes : (+2,+3)=✓, (+3,-2)=✗, (-2,+3)=✗, (+3,+4)=✓ → 2/4 = 0.50

    Args:
        series : Série temporelle d'au moins 3 points.

    Returns:
        Score ∈ [0, 1]. Retourne 0.5 (neutre) si série trop courte.
    """
    n = len(series)
    if n < 3:
        return 0.5

    deltas = [series[i] - series[i - 1] for i in range(1, n)]

    # Produit > 0 ↔ même signe (les deux positifs ou les deux négatifs)
    consistent = sum(1 for i in range(1, len(deltas)) if deltas[i] * deltas[i - 1] > 0)

    return round(consistent / (n - 2), 4)


# 4. Burst score (Kleinberg)
def compute_burst_score(
    series: list[float],
    max_states: int = 2,
) -> list[int]:
    """
    Détection de bursts par automate à états (Kleinberg, 2002).

    Principe :
        On modélise la série comme émise par un automate à (max_states + 1) états.
        Chaque état k correspond à un taux d'émission Poisson différent :
            taux_k = μ × 2^k   (doublement par état, standard dans la littérature)

        L'algorithme de Viterbi trouve la séquence d'états optimale :
        celle qui minimise (coût d'émission + coût de transition).

        Coût de transition vers un état plus élevé = log(n) × (différence d'état),
        ce qui limite les transitions intempestives dans les courtes séries.
        Les transitions vers un état plus bas (retour à la normale) sont gratuites.

    Interprétation :
        état 0 → activité normale
        état 1 → burst de premier niveau (activité 2× au-dessus de la moyenne)
        état 2 → burst intense (activité 4× au-dessus de la moyenne)

    Note : un burst de niveau 1 ou 2 est un candidat signal faible, surtout
    combiné à un emergence_score élevé et un volume absolu faible.

    Args:
        series     : Série temporelle des volumes mensuels.
        max_states : Nombre d'états de burst (défaut : 2).

    Returns:
        Liste d'états (entiers 0…max_states) de même longueur que la série.
        Retourne [0, …] si la série est trop courte ou vide.
    """
    n = len(series)
    if n < 3:
        return [0] * n

    arr = np.array(series, dtype=float)
    mean = arr.mean()
    if mean < 1e-10:
        return [0] * n

    # Taux d'émission pour chaque état (doublement standard)
    rates = [mean * (2.0**k) for k in range(max_states + 1)]

    # Coût de transition proportionnel à log(n) pour contrôler les changements d'état
    transition_unit = math.log(max(n, 2))

    def _emission_cost(count: float, rate: float) -> float:
        """Coût d'émission Poisson : -log P(count | rate) ∝ rate - count·log(rate)."""
        return rate - count * math.log(rate + 1e-10)

    INF = float("inf")
    # cost[t][q] = coût minimal pour être en état q au temps t
    cost = [[INF] * (max_states + 1) for _ in range(n)]
    prev = [[0] * (max_states + 1) for _ in range(n)]

    # Initialisation : tous les états accessibles au temps 0
    for q in range(max_states + 1):
        cost[0][q] = _emission_cost(arr[0], rates[q])

    # Passage avant (Viterbi forward)
    for t in range(1, n):
        for q in range(max_states + 1):
            best_cost = INF
            best_prev = 0
            for q_prev in range(max_states + 1):
                # Monter d'état coûte (q - q_prev) × log(n), descendre est gratuit
                trans_cost = max(0, q - q_prev) * transition_unit
                c = cost[t - 1][q_prev] + trans_cost + _emission_cost(arr[t], rates[q])
                if c < best_cost:
                    best_cost = c
                    best_prev = q_prev
            cost[t][q] = best_cost
            prev[t][q] = best_prev

    # Retraçage arrière (backtrack)
    states = [0] * n
    states[n - 1] = int(np.argmin(cost[n - 1]))
    for t in range(n - 2, -1, -1):
        states[t] = prev[t + 1][states[t + 1]]

    return states
