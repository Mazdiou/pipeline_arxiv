"""
Factory de création du modèle BERTopic complet à partir d'une TopicConfig.

Composants assemblés :
  UMAP             : réduction dimensionnelle (768d : n_components)
  HDBSCAN          : clustering density-based
  CountVectorizer  : vocabulaire c-TF-IDF avec stop words anglais + lemmatisation
  KeyBERTInspired  : labels sémantiques sans redondance via MMR
  BERTopic         : orchestrateur
"""

from __future__ import annotations

import logging
from typing import Any

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..config.topic_settings import TopicConfig

logger = logging.getLogger(__name__)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


class LemmaTokenizer:
    def __init__(self) -> None:
        self.wnl = WordNetLemmatizer()
        self._tokenize = CountVectorizer().build_tokenizer()

    def __call__(self, doc: str) -> list[str]:
        return [
            self.wnl.lemmatize(t)
            for t in self._tokenize(doc)
            if t not in ENGLISH_STOP_WORDS
        ]


def create_bertopic_model(config: TopicConfig, sentence_model: Any = None) -> Any:
    """
    Instancie un modèle BERTopic complet depuis une TopicConfig.

    Args:
        config         : TopicConfig avec tous les paramètres BERTopic.
        sentence_model : SentenceTransformer partagé depuis ArticleEmbedder.

    Returns:
        Instance BERTopic prête pour fit_transform() ou transform().
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_components=config.umap_n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=(
            config.umap_random_state
            if config.umap_random_state is not None
            else config.random_state
        ),
        low_memory=config.umap_low_memory,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric=config.hdbscan_metric,
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        prediction_data=config.hdbscan_prediction_data,
        core_dist_n_jobs=config.hdbscan_core_dist_n_jobs,
    )

    # LemmaTokenizer réduit la redondance (neural / networks : network).
    vectorizer_model = CountVectorizer(
        tokenizer=LemmaTokenizer(),
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        stop_words="english",
    )

    # reduce_frequent_words=True applique une pondération IDF supplémentaire sur
    # les mots très fréquents dans tout le corpus, pour les rendre moins influents dans les labels.
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=config.reduce_frequent_words,
    )

    representation_model = [
        KeyBERTInspired(
            top_n_words=config.representation_top_n_words,
            nr_repr_docs=config.nr_repr_docs,
        ),
        MaximalMarginalRelevance(
            diversity=0.3,
            top_n_words=config.top_n_words,
        ),
    ]

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        embedding_model=sentence_model,
        top_n_words=config.top_n_words,
        nr_topics=config.nr_topics,
        calculate_probabilities=config.calculate_probabilities,
        verbose=False,
    )

    logger.debug(
        "BERTopic instancié : nr_topics=%s | umap_n_neighbors=%d | hdbscan_min_cluster_size=%d",
        config.nr_topics,
        config.umap_n_neighbors,
        config.hdbscan_min_cluster_size,
    )
    return model
