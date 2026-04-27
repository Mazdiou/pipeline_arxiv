"""
Extraction d'entités nommées (NER) dans les articles scientifiques avec spaCy.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ..config.ner_settings import NerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NerResult:
    """
    Résultat de l'extraction NER pour une liste d'articles.
    """

    entities: list

    def __len__(self) -> int:
        return len(self.entities)

    def to_dataframe(self, index: Optional[Any] = None) -> pd.DataFrame:
        """
        Convertit le résultat en DataFrame.

        Returns:
            pd.DataFrame avec la colonne entity_list.
        """
        return pd.DataFrame(
            {"entity_list": pd.array(self.entities, dtype=object)},
            index=index,
        )


class ArticleNerExtractor:
    """
    Extrait les entités nommées scientifiquement pertinentes par article.
    """

    def __init__(self, config: Optional[NerConfig] = None) -> None:
        """
        Args:
            config: Configuration NER (défaut : NerConfig()).
        """
        self._cfg = config or NerConfig()
        self._nlp: Any = None

    # API publique
    def extract(self, texts: list[str]) -> NerResult:
        """
        Extrait les entités nommées pour une liste de textes.
        """
        if not self._cfg.enabled:
            return NerResult(entities=[[] for _ in texts])

        if not texts:
            return NerResult(entities=[])

        self._ensure_model()

        entities: list[list[dict]] = []
        for doc in self._nlp.pipe(texts, batch_size=self._cfg.batch_size):
            entities.append(self._process_doc(doc))

        return NerResult(entities=entities)

    # Méthodes internes
    def _ensure_model(self) -> None:
        """Charge le modèle spaCy en lazy si ce n'est pas encore fait."""
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spacy est requis pour le NER. "
                "Installation : pip install spacy scispacy && "
                f"pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{self._cfg.model_name}-0.5.4.tar.gz"
            ) from exc

        logger.info("Chargement modèle spaCy : %s", self._cfg.model_name)
        try:
            self._nlp = spacy.load(
                self._cfg.model_name,
                disable=self._cfg.disable_pipes,
            )
        except OSError as exc:
            raise OSError(f"Modèle spaCy '{self._cfg.model_name}' non trouvé.") from exc
        logger.info("Modèle spaCy chargé.")

    def _process_doc(self, doc: Any) -> list[dict]:
        """
        Extrait, déduplique et compte les entités d'un document spaCy.
        """
        blacklist = {t.upper() for t in self._cfg.blacklist_tokens}
        entity_counts: Counter = Counter()
        for ent in doc.ents:
            if ent.label_ not in self._cfg.entity_types:
                continue
            text = ent.text.strip()
            if len(text) < self._cfg.min_entity_length:
                continue
            if text.upper() in blacklist:
                continue
            entity_counts[(text, ent.label_)] += 1

        top = entity_counts.most_common(self._cfg.max_entities)
        return [
            {"text": text, "label": label, "count": count}
            for (text, label), count in top
        ]
