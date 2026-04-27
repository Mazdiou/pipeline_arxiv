"""
Configuration du module de topic modeling (BERTopic).

TopicConfig expose tous les paramètres clés de BERTopic organisés en sections :
  1. Cycle de vie des modèles  (persistance, refit, seuils)
  2. UMAP                      (réduction dimensionnelle : n_components)
  3. HDBSCAN                   (clustering density-based)
  4. BERTopic                  (orchestration globale)
  5. CountVectorizer           (vocabulaire c-TF-IDF)
  6. Représentation            (labels de topics via KeyBERTInspired)
  7. Réduction des outliers    (réassignation des articles non classés)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TopicConfig:
    """
    Configuration complète du module ArticleTopicModeler.
    """

    enabled: bool = True
    """
    False : désactive entièrement BERTopic. Tous les articles reçoivent
            topic_id=-1, topic_label='outlier', topic_prob=0.0.
    """

    # Reproductibilité globale
    random_state: int = 42
    """
    Graine aléatoire globale utilisée pour toutes les étapes stochastiques de BERTopic.
    """

    # Cycle de vie des modèles
    models_dir: str = "models/topics"
    """
    Dossier racine pour les modèles BERTopic sérialisés par catégorie.
    """

    force_refit: bool = False
    """
    False : Utilise le modèle BERTopic existant pour transform() si disponible.
            Fit initial automatique si aucun modèle n'existe encore.
            
    True  : Ignore le modèle existant et réentraîne depuis zéro sur TOUTES
            les partitions CLEAN disponibles pour la catégorie.
    """

    min_articles: int = 700
    """
    Nombre minimum d'articles pour déclencher un fit BERTopic.
    En dessous du seuil : tous les articles reçoivent topic_id=-1 (outlier).
    """

    # UMAP : Réduction dimensionnelle
    umap_n_components: int = 8
    """
    Nombre de dimensions de l'espace UMAP (input de HDBSCAN).
    """

    umap_n_neighbors: int = 25
    """
    Taille du voisinage local pour UMAP (structure locale vs. globale).
    """

    umap_min_dist: float = 0.0
    """
    Distance minimale entre points dans l'espace UMAP réduit.
    """

    umap_metric: str = "cosine"
    """
    Métrique de distance utilisée par UMAP sur les embeddings bruts.
    """

    umap_random_state: int | None = None
    """
    Graine aléatoire spécifique à UMAP. Si None, utilise la valeur de random_state.
    """

    umap_low_memory: bool = False
    """
    True  : algorithme UMAP approximatif (NN-descent).
    False : algorithme exact, plus précis.
    """

    # HDBSCAN : Clustering hiérarchique density-based
    hdbscan_min_cluster_size: int = 15
    """
    Taille minimale d'un cluster HDBSCAN.
    """

    hdbscan_min_samples: int = 3
    """
    Densité locale minimale pour qu'un point soit considéré 'core point'.
    """

    scale_cluster_size: bool = True
    """
    True : hdbscan_min_cluster_size est recalculé dynamiquement à partir de la taille
    du corpus au moment du fit.
    """

    cluster_size_divisor: int = 150
    """
    Diviseur utilisé pour le calcul dynamique de hdbscan_min_cluster_size (si scale_cluster_size=True).
    Formule : max(hdbscan_min_cluster_size, min(50, n // cluster_size_divisor))
    """

    scale_n_neighbors: bool = True
    """
    True : umap_n_neighbors est recalculé dynamiquement.
    """

    n_neighbors_divisor: int = 150
    """
    Diviseur pour le calcul dynamique de umap_n_neighbors (si scale_n_neighbors=True).
    Formule : max(umap_n_neighbors, min(50, n // n_neighbors_divisor))
    """

    hdbscan_metric: str = "euclidean"
    """
    Métrique HDBSCAN dans l'espace UMAP réduit.
    """

    hdbscan_cluster_selection_method: str = "eom"
    """
    Méthode de sélection des clusters HDBSCAN.
    'eom'  : Excess of Mass : clusters de taille variable, adaptatif.
    'leaf' : Clusters feuilles de la hiérarchie : topics plus petits et nombreux.
    """

    hdbscan_prediction_data: bool = True
    """
    True : Pré-calcule les structures de données pour transform() sur new data.
           Indispensable pour prédire les topics de nouvelles partitions.
    """

    hdbscan_core_dist_n_jobs: int = -1
    """
    Nombre de threads pour le calcul des core distances HDBSCAN.
    -1 : utilise tous les cœurs CPU disponibles.
     1 : mono-threadé (reproductibilité stricte, débogage).
    """

    # BERTopic : Paramètres principaux
    top_n_words: int = 30
    """
    Nombre de mots représentatifs extraits par topic (via c-TF-IDF).
    """

    nr_topics: int | str | None = None
    """
    Nombre cible de topics après fusion automatique.
    None : BERTopic détermine le nombre optimal automatiquement.
    Entier N : Force la fusion des topics similaires jusqu'à N topics.
    """

    calculate_probabilities: bool = False
    """
    True  : Calcule les probabilités d'appartenance à TOUS les topics pour
            chaque document (matrice n × n_topics).
    """

    # CountVectorizer + ClassTfidfTransformer : Vocabulaire c-TF-IDF
    ngram_range: tuple[int, int] = (1, 2)
    """
    Plage de n-grammes pour le vocabulaire c-TF-IDF.
    """

    min_df: int = 1
    """
    Fréquence minimale d'un terme dans le corpus pour entrer dans le vocabulaire.
    """

    reduce_frequent_words: bool = True
    """
    True : Applique un filtrage supplémentaire pour réduire les mots très fréquents.
    """

    # Représentation : Labels de topics (KeyBERTInspired)
    representation_top_n_words: int = 60
    """
    Nombre de mots candidats envoyés à KeyBERTInspired pour le re-ranking.
    """

    nr_repr_docs: int = 25
    """
    Nombre de documents représentatifs par topic utilisés par KeyBERTInspired
    pour calculer la similarité sémantique avec les mots candidats.
    """

    # Réduction des outliers
    outlier_threshold: float = 0.30
    """
    Seuil de similarité cosinus minimale pour réassigner un outlier à un topic.
    """

    # Dynamic topic modelling : drift-triggered refit
    refit_outlier_threshold: float = 0.25
    """
    Taux d'outliers au-delà duquel un re-fit automatique est déclenché.
    """

    refit_window_months: int = 12
    """
    Nombre de partitions CLEAN les plus récentes utilisées pour le re-fit.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.min_articles < 1:
            raise ValueError("min_articles doit être >= 1.")
        if self.umap_n_components < 2:
            raise ValueError("umap_n_components doit être >= 2.")
        if self.umap_n_neighbors < 2:
            raise ValueError("umap_n_neighbors doit être >= 2.")
        if not (0.0 <= self.umap_min_dist <= 1.0):
            raise ValueError("umap_min_dist doit être dans [0.0, 1.0].")
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError("hdbscan_min_cluster_size doit être >= 2.")
        if self.hdbscan_min_samples < 1:
            raise ValueError("hdbscan_min_samples doit être >= 1.")
        if self.hdbscan_min_samples > self.hdbscan_min_cluster_size:
            raise ValueError(
                f"hdbscan_min_samples ({self.hdbscan_min_samples}) ne peut pas être "
                f"supérieur à hdbscan_min_cluster_size ({self.hdbscan_min_cluster_size})."
            )
        if self.cluster_size_divisor < 1:
            raise ValueError("cluster_size_divisor doit être >= 1.")
        if self.n_neighbors_divisor < 1:
            raise ValueError("n_neighbors_divisor doit être >= 1.")
        if self.hdbscan_cluster_selection_method not in ("eom", "leaf"):
            raise ValueError(
                f"hdbscan_cluster_selection_method invalide : "
                f"'{self.hdbscan_cluster_selection_method}'. "
                "Valeurs acceptées : 'eom', 'leaf'."
            )
        if self.top_n_words < 1:
            raise ValueError("top_n_words doit être >= 1.")
        if self.min_df < 1:
            raise ValueError("min_df doit être >= 1.")
        if self.nr_repr_docs < 1:
            raise ValueError("nr_repr_docs doit être >= 1.")
        if self.representation_top_n_words < self.top_n_words:
            raise ValueError("representation_top_n_words doit être >= top_n_words.")
        if not (0.0 <= self.outlier_threshold <= 1.0):
            raise ValueError("outlier_threshold doit être dans [0.0, 1.0].")
        if not (0.0 < self.refit_outlier_threshold <= 1.0):
            raise ValueError("refit_outlier_threshold doit être dans (0.0, 1.0].")
        if self.refit_window_months < 1:
            raise ValueError("refit_window_months doit être >= 1.")
