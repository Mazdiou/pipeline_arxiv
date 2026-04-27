"""
TopicLabels : génère des labels textuels pour les topics BERTopic.
- Utilise les mots-clés BERTopic (get_topics()) pour construire des labels concis.
- Fallback sur le champ Name de get_topic_info() si get_topics() est indisponible.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from .topic_constants import STEMMER, LABEL_STOP_WORDS

logger = logging.getLogger(__name__)


def extract_labels(model: Any, topics: list[int], texts=None) -> list[str]:
    """
    Convertit topic_id → label texte.
    Fallback chain : get_topics() → Name BERTopic → topic_{id} → misc
    """
    topic_map: dict[int, str] = {}

    # Niveau 1 : labels depuis les mots-clés BERTopic
    try:
        for tid, words_scores in model.get_topics().items():
            if tid != -1:
                topic_map[tid] = _build_label(words_scores, tid)
    except Exception as exc:
        logger.warning("get_topics() échoué (%s) — fallback sur Name", exc)

    # Niveau 2 : fallback sur le champ Name BERTopic ("42_neural_network_learning")
    try:
        name_map = model.get_topic_info().set_index("Topic")["Name"].to_dict()
        for tid, name in name_map.items():
            if tid == -1:
                continue
            current = topic_map.get(tid, "")
            if not current or current.startswith("topic_"):
                cleaned = _clean_name(str(name), tid)
                if not cleaned.startswith("topic_"):
                    topic_map[tid] = cleaned
    except Exception as exc:
        logger.warning("get_topic_info() échoué (%s) — labels dégradés", exc)

    labels = []
    for i, tid in enumerate(topics):
        if tid == -1:
            text = texts[i] if texts and i < len(texts) else None
            labels.append(_label_from_text(text) if text else "misc")
        else:
            labels.append(topic_map.get(tid, f"topic_{tid}"))

    return labels


def _stems_collide(stem: str, seen_stems: set[str]) -> bool:
    """True si stem est morphologiquement trop proche d'un stem déjà vu (préfixe 4 chars)."""
    return any(stem.startswith(s[:4]) or s.startswith(stem[:4]) for s in seen_stems)


def _build_label(words_scores, topic_id, max_terms=3) -> str:
    """Construit un label depuis les mots BERTopic, en filtrant stopwords et stems redondants."""
    clean_terms = []
    seen_stems = set()

    for word, _ in words_scores[:40]:
        parts = word.lower().split()
        if len(word) < 2 or all(p in LABEL_STOP_WORDS for p in parts):
            continue
        stem = STEMMER.stem(parts[0])
        if _stems_collide(stem, seen_stems):
            continue
        seen_stems.add(stem)
        clean_terms.append("_".join(parts))
        if len(clean_terms) >= max_terms:
            break

    # Fallback : premier mot non-stop
    if not clean_terms:
        for word, _ in words_scores:
            for p in word.lower().split():
                if p not in LABEL_STOP_WORDS:
                    clean_terms.append(p)
                    break
            if clean_terms:
                break

    return "_".join(clean_terms) if clean_terms else f"topic_{topic_id}"


def _clean_name(raw_name: str, topic_id: int) -> str:
    """Nettoie un label issu du champ Name BERTopic ("42_neural_network" → "neural_network")."""
    parts = raw_name.split("_", 1)
    label_part = parts[1] if len(parts) == 2 and parts[0].isdigit() else raw_name
    tokens = [
        t for t in label_part.split("_") if t and t.lower() not in LABEL_STOP_WORDS
    ]
    return "_".join(tokens[:3]) if tokens else f"topic_{topic_id}"


def _label_from_text(text: str, max_terms=3) -> str:
    """Génère un label depuis le texte brut d'un article outlier (mots les plus fréquents)."""
    words = [
        w
        for w in text.lower().split()
        if len(w) >= 4 and w.isalpha() and w not in LABEL_STOP_WORDS
    ]
    if not words:
        return "misc"

    clean = []
    seen_stems = set()
    for w, _ in Counter(words).most_common(max_terms * 4):
        stem = STEMMER.stem(w)
        if _stems_collide(stem, seen_stems):
            continue
        seen_stems.add(stem)
        clean.append(w)
        if len(clean) >= max_terms:
            break

    return "_".join(clean) if clean else "misc"
