"""
Configuration du module d'embeddings sémantiques.

Choix de modèles recommandés pour la veille scientifique arXiv :

    allenai-specter         768 dims  Spécialisé papiers scientifiques.
                                      Entraîné sur corpus Semantic Scholar.

    all-MiniLM-L6-v2        384 dims  Rapide, léger. Bon pour prototypage
                                      ou machines sans GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingsConfig:
    """
    Paramètres du module ArticleEmbedder.
    """

    model_name: str = "all-MiniLM-L6-v2"
    """
    Identifiant HuggingFace Hub du modèle SentenceTransformer.
    Voir les recommandations en en-tête de module.
    """

    batch_size: int = 32
    """
    Nombre d'articles encodés simultanément.
    Augmenter si GPU disponible (64–256). Réduire si OOM.
    """

    normalize_embeddings: bool = True
    """
    Normalisation L2 des vecteurs (norme = 1).

    True  → similarité cosinus = produit scalaire = calculs plus rapides.
    False → vecteurs bruts.
    """

    show_progress_bar: bool = True
    """
    Afficher la barre de progression tqdm lors de l'encoding.
    """

    store_in_gold: bool = False
    """
    Persister les embeddings dans le fichier GOLD Parquet.

    False → calculés en mémoire, partagés entre les modules NLP, non écrits.
    True  → écrits dans la colonne 'embedding' du GOLD.
    """

    device: Optional[str] = None
    """
    Dispositif de calcul pour SentenceTransformer.

    None  → détection automatique : CUDA > MPS (Apple Silicon) > CPU.
    'cpu' → force le CPU (reproductibilité, débogage).
    'cuda'→ GPU NVIDIA (batch_size peut être augmenté).
    'mps' → GPU Apple Silicon (batch_size modéré recommandé).
    """

    cache_dir: Optional[str] = "models/sentence_transformers"
    """
    Répertoire local de cache des modèles HuggingFace.

    None → répertoire par défaut HuggingFace (~/.cache/huggingface/).
    Chemin → ex: 'models/sentence_transformers' pour environnements
             sans accès internet (modèle pré-téléchargé).
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.model_name or not self.model_name.strip():
            raise ValueError("'model_name' ne peut pas être vide.")

        if self.batch_size <= 0:
            raise ValueError(f"'batch_size' doit être > 0 (reçu : {self.batch_size})")

        valid_devices = {"cpu", "cuda", "mps"}
        if self.device is not None and self.device not in valid_devices:
            raise ValueError(
                f"'device' invalide : '{self.device}'. "
                f"Valeurs acceptées : {sorted(valid_devices)} ou None (auto)."
            )

    def resolved_device(self) -> str:
        """
        Retourne le dispositif effectif à utiliser.

        Si device=None, détecte automatiquement dans l'ordre :
            CUDA (GPU NVIDIA) > MPS (Apple Silicon) > CPU.

        Returns:
            'cuda' | 'mps' | 'cpu'
        """
        if self.device is not None:
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"
