# SignalScorer documentation des métriques et du scoring

## Objectif

Le `SignalScorer` a pour but de transformer un ensemble de métriques hétérogènes en un score composite permettant de :

- détecter des **tendances fortes**
- identifier des **signaux faibles émergents**
- mesurer des **transformations sémantiques**

Le système repose sur 3 dimensions principales :

1. **Trend (tendance)**
2. **Weak Signal (signal faible)**
3. **Semantic (transformation sémantique)**

---

## Vue d’ensemble du pipeline

```
Métriques brutes
      ↓
Normalisation (sigmoïde / scaling)
      ↓
Agrégation par dimension
      ↓
Score composite pondéré
```

---

# Trend Score : Tendance volumétrique

## 🎯 Objectif

Mesurer la **force, direction et stabilité** d’une tendance dans le temps.

---

## Métriques utilisées

### 1. Z-score (`z_score_last`)

**Définition :**
Mesure à quel point la valeur actuelle est atypique par rapport à l’historique.

**Interprétation :**

- > 0 → au-dessus de la moyenne
- > 0 → pic anormal (potentielle explosion)

**Impact :**

- Détecte les **accélérations soudaines**
- Sensible aux anomalies

---

### 2. OLS Slope (`ols_slope`)

**Définition :**
Pente d’une régression linéaire sur la série temporelle.

**Interprétation :**

- > 0 → croissance
- > < 0 → décroissance

**Impact :**

- Mesure la **vitesse de croissance**
- Plus robuste que le z-score

---

### 3. Mann-Kendall Tau (`mann_kendall_tau`)

**Définition :**
Statistique non paramétrique mesurant une tendance monotone.

**Intervalle :**
[-1, 1]

**Interprétation :**

- 1 : tendance strictement croissante
- 0 : pas de tendance
- -1 : tendance décroissante

**Impact :**

- Capture la **direction globale**
- Robuste au bruit

---

### 4. Hurst Exponent (`hurst_exponent`)

**Définition :**
Mesure la mémoire long terme d’une série.

**Intervalle :**
[0, 1]

**Interprétation :**

- > 0.5 → tendance persistante
- > < 0.5 → comportement erratique

**Impact :**

- Différencie une vraie tendance d’un bruit aléatoire

---

### 5. CAGR (`cagr`)

**Définition :**
Taux de croissance annuel composé.

**Impact :**

- Mesure la **croissance globale**
- Complète la pente locale

---

## Logique de calcul

- Normalisation via sigmoïde pour les valeurs non bornées
- Moyenne simple des composantes

```
trend_score = moyenne(
    z_score,
    slope,
    tau,
    hurst,
    cagr
)
```

---

## Interprétation

| Score | Signification                |
| ----- | ---------------------------- |
| ~0.2  | Pas de tendance              |
| ~0.5  | Tendance faible/modérée      |
| ~0.8+ | Tendance forte et structurée |

---

# Weak Signal Score : Signal faible

## Objectif

Détecter des **phénomènes émergents précoces**, avant qu’ils deviennent des tendances.

---

## Métriques utilisées

### 1. Burst State (`burst_state_last`)

**Définition :**
Niveau d’intensité d’activité (modèle de burst).

**Normalisation :**

```
burst_state / burst_max_states
```

**Impact :**

- Détecte des **pics soudains**
- Indicateur d’attention récente

---

### 2. Emergence Score (`emergence_score_last`)

**Définition :**
Mesure la croissance récente.

**Impact :**

- Capture les **démarrages rapides**
- Sensible aux early signals

---

### 3. Novelty Weighted Emergence (`novelty_weighted_emergence_last`)

**Définition :**
Emergence pondérée par la nouveauté.

**Impact :**

- Favorise les sujets **nouveaux + en croissance**
- Filtre les tendances déjà établies

---

### 4. Persistence Score (`persistence_score`)

**Définition :**
Mesure la continuité du signal dans le temps.

**Intervalle :**
[0, 1]

**Impact :**

- Évite les faux positifs (pics isolés)
- Renforce les signaux durables

---

## Logique de calcul

```
weak_signal_score = moyenne(
    burst,
    emergence,
    nwe,
    persistence
)
```

---

## Interprétation

| Score | Signification                     |
| ----- | --------------------------------- |
| ~0.2  | Pas de signal                     |
| ~0.5  | Signal émergent                   |
| ~0.8+ | Signal faible fort (pré-tendance) |

---

# Semantic Score : Transformation sémantique

## Objectif

Mesurer **à quel point le contenu d’un topic évolue**.

Important : ce score mesure le **changement**, pas la qualité.

---

## Métriques utilisées

### 1. Semantic Drift (`semantic_drift_last`)

**Définition :**
Changement global de signification du contenu.

**Impact :**

- Détecte les **évolutions de sens**
- Utile pour identifier des pivots

---

### 2. Keyword Drift (`keyword_drift_last`)

**Définition :**
Évolution du vocabulaire utilisé.

**Impact :**

- Capture les changements lexicaux
- Indique une mutation du discours

---

### 3. Stability Score (`stability_score`)

**Définition :**
Mesure de stabilité globale.

**Transformation :**

```
instability = 1 - stability
```

**Impact :**

- Plus c’est instable : plus le score augmente
- Capture les phases de transformation

---

## Logique de calcul

```
semantic_score = moyenne(
    drift,
    keyword_drift,
    instability
)
```

---

## Interprétation

| Score | Signification        |
| ----- | -------------------- |
| ~0.2  | Stable               |
| ~0.5  | Évolution modérée    |
| ~0.8+ | Transformation forte |

---

# Score Composite

## Objectif

Fournir un score global combinant les 3 dimensions.

---

## Formule

```
composite_score = (α * trend_score + β * weak_signal_score +
                γ * semantic_score) / (α + β + γ)
```

---

## Pondérations par défaut

| Dimension   | Poids |
| ----------- | ----- |
| Trend       | 0.4   |
| Weak Signal | 0.4   |
| Semantic    | 0.2   |

---

## Interprétation

| Score  | Signification     |
| ------ | ----------------- |
| ~0.3   | Faible intérêt    |
| ~0.5   | Sujet intéressant |
| ~0.7+  | Sujet prioritaire |
| ~0.85+ | Signal critique   |
