"""
Configuration du module de Named Entity Recognition (NER) avec spaCy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class NerConfig:
    """
    Paramètres du module ArticleNerExtractor.
    """

    # Modèle spaCy
    model_name: str = "en_core_sci_lg"
    """
    Modèle spaCy à charger.
    
    Options spaCy :
      - en_core_web_trf : transformer-based (≈400 MB), haute précision
      - en_core_sci_lg  : large, vecteurs scientifiques inclus
    """

    disable_pipes: List[str] = field(
        default_factory=lambda: ["parser", "lemmatizer", "attribute_ruler"]
    )
    """
    Composants spaCy à désactiver pour accélérer le traitement.
    Note scispaCy : attribute_ruler peut être absent — ignoré silencieusement par spaCy.
    """

    # Filtrage des entités
    entity_types: List[str] = field(
        default_factory=lambda: [
            "ENTITY",
            "PERSON",
            "ORG",
            "GPE",
            "PRODUCT",
            "WORK_OF_ART",
        ]
    )
    """
    Types d'entités à conserver dans la sortie (filtrage post-NER).
    "ENTITY" est le label générique de scispaCy pour les entités scientifiques.
    """

    max_entities: int = 5
    """
    Nombre maximum d'entités retenues par article (triées par fréquence décroissante).
    """

    min_entity_length: int = 2
    """
    Longueur minimale du texte d'une entité pour être conservée.
    """

    blacklist_tokens: List[str] = field(default_factory=lambda: ["MATH"])
    """
    Tokens à exclure des entités détectées.
    """

    # Traitement
    batch_size: int = 32
    """
    Taille des lots pour spaCy nlp.pipe() (traitement en batch des textes).
    """

    enabled: bool = True
    """
    False : NER désactivé, entity_list = [] pour tous les articles.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.max_entities < 1:
            raise ValueError(
                f"'max_entities' doit être >= 1 (reçu : {self.max_entities})"
            )
        if self.batch_size < 1:
            raise ValueError(f"'batch_size' doit être >= 1 (reçu : {self.batch_size})")
        if self.min_entity_length < 1:
            raise ValueError(
                f"'min_entity_length' doit être >= 1 (reçu : {self.min_entity_length})"
            )
        if not self.model_name.strip():
            raise ValueError("'model_name' ne peut pas être vide.")
