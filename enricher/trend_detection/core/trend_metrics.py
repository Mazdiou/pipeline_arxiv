"""
Métriques statistiques de tendance sur des séries temporelles de volume.

Chaque fonction est pure : list[float] → scalaire ou liste.
Aucune dépendance sur les classes métier du pipeline.

    1. Z-score glissant  : anomalie locale par rapport à l'historique récent
    2. Pente OLS         : croissance linéaire globale
    3. Mann-Kendall      : tendance monotone non-paramétrique, robuste aux outliers
    4. Exposant de Hurst : persistance long-terme
    5. CAGR              : taux de croissance annualisé, comparaison inter-topics
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# 1. Z-score glissant
def compute_z_score_series(series: list[float], window: int = 6) -> list[float]:
    """
    Quantifie le caractère exceptionnel de chaque mois par rapport à son historique récent.

    Pour chaque point t > ``window`` :

        z_t = (x_t - μ_fenêtre) / σ_fenêtre

    Seuils d'interprétation indicatifs :

        |z| < 1.0  : dans la plage habituelle
        z > +2.0   : pic anormal (top ~2,5 % de la distribution de la fenêtre)
        z < -2.0   : creux anormal

    Les ``window`` premiers points sont mis à 0.0 (historique insuffisant).
    Si σ = 0 (tous les points de la fenêtre sont identiques), la valeur courante
    est traitée comme un écart infini et codée ±3.0 selon sa direction.

    Paramètres :

    series : list[float]
        Volumes mensuels (nombre d'articles par mois).
    window : int
        Taille de la fenêtre glissante en mois (défaut : 6).

    Retourne :

    list[float]
        Série de z-scores de même longueur que l'entrée.
    """
    n = len(series)
    result = [0.0] * n

    for i in range(window, n):
        past = series[i - window : i]
        mu = float(np.mean(past))
        sigma = float(np.std(past, ddof=1))

        if sigma > 1e-10:
            result[i] = round((series[i] - mu) / sigma, 4)
        elif abs(series[i] - mu) < 1e-10:
            result[i] = 0.0
        else:
            # Fenêtre constante mais valeur courante différente :
            # l'écart est réel mais non quantifiable : valeur sentinelle ±3.0.
            result[i] = 3.0 if series[i] > mu else -3.0

    return result


# 2. Pente OLS
def compute_ols_slope(series: list[float]) -> float:
    """
    Estime la croissance linéaire globale par régression OLS.

    Modèle ajusté : y = a·t + b, avec t = 0, 1, 2, …

    Interprétation de la pente ``a`` :

        a > 0  : croissance structurelle (articles supplémentaires par mois)
        a < 0  : déclin linéaire
        a ≈ 0  : volume stable

    Limite : la pente OLS suppose une relation linéaire. Pour des tendances
    exponentielles ou en plateau, préférer Mann-Kendall qui est non-paramétrique.

    Paramètres :

    series : list[float]
        Série temporelle (au moins 2 points requis).

    Retourne :

    float
        Pente en articles/mois. Retourne 0.0 si la série est trop courte.
    """
    n = len(series)
    if n < 2:
        return 0.0

    x = np.arange(n, dtype=float)
    y = np.array(series, dtype=float)
    slope, *_ = stats.linregress(x, y)
    return round(float(slope), 6)


# 3. Test de Mann-Kendall
def compute_mann_kendall(series: list[float]) -> tuple[float, float]:
    """
    Détecte une tendance monotone sans hypothèse sur la distribution des données.

    Principe : pour chaque paire (i < j), le test vote +1 si la série monte,
    -1 si elle descend. La statistique tau résume ces votes.
    Implémenté via ``scipy.kendalltau(t, série)``, équivalent au test M-K classique.

    Avantages sur OLS :

        - Non-paramétrique : aucune hypothèse de normalité
        - Robuste aux outliers et aux plateaux prolongés
        - Détecte les tendances non-linéaires (exponentielle, sigmoïde…)

    Paramètres :

    series : list[float]
        Série temporelle sur mois actifs uniquement (au moins 4 points requis).

    Retourne :

    tuple[float, float]
        ``(tau, p_value)`` où :

        - ``tau`` ∈ [-1, 1] : force et direction de la tendance
          (positif = croissance, négatif = déclin)
        - ``p_value`` ∈ [0, 1] : probabilité d'observer ce tau sous H₀ (absence
          de tendance) ; < 0.05 indique une tendance statistiquement significative

        Retourne ``(0.0, 1.0)`` si la série est trop courte.
    """
    if len(series) < 4:
        return 0.0, 1.0

    time_index = np.arange(len(series), dtype=float)
    result = stats.kendalltau(time_index, series)
    return round(float(result.statistic), 4), round(float(result.pvalue), 4)


# 4. Exposant de Hurst
def compute_hurst_exponent(series: list[float], min_length: int = 8) -> float:
    """
    Mesure la persistance d'une série : une hausse tend-elle à en appeler une autre ?

    Méthode R/S (Rescaled Range, Hurst 1951) : pour plusieurs tailles de fenêtre n,
    on calcule le rapport R/S entre l'étendue des déviations cumulées (R) et l'écart-
    type de la fenêtre (S). La régression log(R/S) ~ log(n) donne la pente H.

    Interprétation de H :

        H > 0.5  : série persistante — les hausses ont tendance à se prolonger
        H = 0.5  : marche aléatoire — le futur est indépendant du passé
        H < 0.5  : anti-persistante — alternances hausse/baisse (bruit oscillatoire)

    Complémentarité avec Mann-Kendall :
        Mann-Kendall indique la *direction* de la tendance.
        Hurst valide qu'elle *persiste* dans le temps et n'est pas du bruit.
        Un topic avec tau ≥ 0.40 ET H ≥ 0.55 est un signal TRENDING très fiable.

    Paramètres :

    series : list[float]
        Série temporelle sur mois actifs uniquement.
    min_length : int
        Longueur minimale requise pour le calcul (défaut : 8).

    Retourne :

    float
        H ∈ [0, 1]. Retourne 0.5 (neutre) si la série est trop courte
        ou si la régression R/S manque de points pour être fiable.
    """
    n = len(series)
    if n < min_length:
        return 0.5

    arr = np.array(series, dtype=float)
    rs_pairs: list[tuple[int, float]] = []

    # Fenêtres de taille 3 à n//2 pour couvrir plusieurs échelles temporelles.
    for window in range(3, n // 2 + 1):
        rs_for_window: list[float] = []
        for start in range(0, n - window + 1, window):
            segment = arr[start : start + window]
            mean = segment.mean()
            deviations = np.cumsum(segment - mean)
            R = deviations.max() - deviations.min()
            S = segment.std(ddof=1)
            if S > 1e-10:
                rs_for_window.append(R / S)
        if rs_for_window:
            rs_pairs.append((window, float(np.mean(rs_for_window))))

    # Trop peu de tailles de fenêtre : la régression log-log serait instable.
    if len(rs_pairs) < 4:
        return 0.5

    log_sizes = np.log([p[0] for p in rs_pairs])
    log_rs = np.log([p[1] for p in rs_pairs])
    slope, _ = np.polyfit(log_sizes, log_rs, 1)

    return round(float(np.clip(slope, 0.0, 1.0)), 4)


# 5. CAGR
def compute_cagr(series: list[float]) -> float:
    """
    Calcule le taux de croissance annuel composé (Compound Annual Growth Rate).

        CAGR = (V_final / V_initial) ^ (12 / n_mois) - 1

    Ramène la croissance sur une base annuelle, ce qui permet de comparer
    des topics avec des historiques de durées différentes.

    Exemple :

        V_initial = 10, V_final = 40, sur 24 mois → CAGR ≈ +41 %/an
        V_initial = 10, V_final = 40, sur 12 mois → CAGR = +300 %/an

    Limite : le CAGR ne tient compte que des deux extrémités de la série.
    Il ignore la volatilité intermédiaire : une série oscillante peut afficher
    le même CAGR qu'une série régulièrement croissante.

    ``V_initial`` et ``V_final`` sont définis comme le premier et le dernier
    point *non nul* de la série, pour éviter une division par zéro sur les
    topics inactifs en début ou fin de période.

    Paramètres :

    series : list[float]
        Série temporelle (au moins 2 points non nuls requis).

    Retourne :

    float
        CAGR en proportion (ex. 0.41 = +41 %/an).
        Retourne 0.0 si la série est trop courte ou entièrement nulle.
    """
    n = len(series)
    if n < 2:
        return 0.0

    first = next((v for v in series if v > 0), None)
    last = next((v for v in reversed(series) if v > 0), None)

    if first is None or last is None:
        return 0.0

    years = n / 12.0
    cagr = (last / first) ** (1.0 / years) - 1.0
    return round(float(cagr), 4)
