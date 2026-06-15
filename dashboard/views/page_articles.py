from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from dashboard.core.db import (
    get_article_detail,
    get_articles_count,
    get_articles_page,
    get_categories_with_data,
    get_distinct_langs,
    get_topics_for_category,
)
from dashboard.core.ui_helpers import render_category_selector

_PAGE_SIZE = 25

_NER_LABELS: dict[str, str] = {
    "PERSON": "Personnes",
    "ORG": "Organisations",
    "GPE": "Lieux",
    "LOC": "Lieux",
    "DISEASE": "Maladies",
    "CHEMICAL": "Substances",
    "GENE_OR_GENE_PRODUCT": "Gènes / Protéines",
    "CELL_TYPE": "Types cellulaires",
    "DNA": "ADN",
    "RNA": "ARN",
    "WORK_OF_ART": "Œuvres",
    "EVENT": "Événements",
    "DATE": "Dates",
    "CARDINAL": "Nombres",
}


def render() -> None:
    st.title("Articles")

    categories = get_categories_with_data()
    if not categories:
        st.info(
            "Aucune catégorie disponible. "
            "Configurez et bootstrappez une catégorie depuis la page **Contrôle**.",
            icon="ℹ️",
        )
        return

    category_id = render_category_selector(categories)
    if not category_id:
        return

    if st.session_state.get("_articles_category") != category_id:
        st.session_state["_articles_category"] = category_id
        st.session_state["articles_page"] = 0
        st.session_state.pop("selected_arxiv_id", None)

    filters = _render_filters(category_id)
    selected_id = _render_table(category_id, filters)

    if selected_id:
        st.divider()
        detail = get_article_detail(selected_id, category_id)
        if detail:
            _render_article_detail(detail)


def _render_filters(category_id: str) -> dict:
    topics = get_topics_for_category(category_id)
    langs = get_distinct_langs(category_id)

    with st.expander("Filtres", expanded=True):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

        with col1:
            topic_options = [None] + [t["topic_id"] for t in topics]
            topic_labels = {t["topic_id"]: t["topic_label"] for t in topics}

            def _fmt_topic(tid):
                if tid is None:
                    return "Tous les topics"
                lbl = topic_labels.get(tid, f"Topic {tid}")
                return lbl[:60] + "…" if len(lbl) > 60 else lbl

            selected_topic = st.selectbox(
                "Topic",
                options=topic_options,
                format_func=_fmt_topic,
                key="filter_topic",
            )

        with col2:
            lang_options = ["Toutes"] + langs
            selected_lang = st.selectbox(
                "Langue",
                options=lang_options,
                key="filter_lang",
            )

        with col3:
            novelty_min = st.slider(
                "Nouveauté min.",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="filter_novelty",
                help=(
                    "Filtre par score d'isolement sémantique (0 → 1). "
                    "0 = très similaire aux articles existants du topic, "
                    "1 = très différent (potentiellement nouveau angle de recherche). "
                    "Calculé à partir des embeddings par le module Enricher. "
                    "Recommandé : laisser à 0 pour voir tous les articles, "
                    "monter à 0.5+ pour ne garder que les plus originaux."
                ),
            )

        with col4:
            search = st.text_input(
                "Recherche dans le titre",
                placeholder="ex: transformer",
                key="filter_search",
            )

        col_from, col_to, col_reset = st.columns([2, 2, 1])
        with col_from:
            month_from = st.text_input(
                "Mois début (YYYY-MM)",
                placeholder="ex: 2023-01",
                key="filter_month_from",
            )
        with col_to:
            month_to = st.text_input(
                "Mois fin (YYYY-MM)",
                placeholder="ex: 2024-12",
                key="filter_month_to",
            )
        with col_reset:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Réinitialiser", use_container_width=True):
                for k in (
                    "filter_topic",
                    "filter_lang",
                    "filter_novelty",
                    "filter_search",
                    "filter_month_from",
                    "filter_month_to",
                    "filter_exclude_outliers",
                    "filter_exclude_short",
                ):
                    st.session_state.pop(k, None)
                st.session_state["articles_page"] = 0
                st.session_state.pop("selected_arxiv_id", None)
                st.rerun()

        col_excl1, col_excl2, _ = st.columns([2, 2, 5])
        with col_excl1:
            exclude_outliers = st.checkbox(
                "Exclure les outliers BERTopic",
                key="filter_exclude_outliers",
                help=(
                    "Exclut les articles que BERTopic n'a pas pu assigner à un topic "
                    "(probabilité d'appartenance trop faible). "
                    "Ces articles ne sont pas de mauvaise qualité — ils sont simplement "
                    "trop atypiques pour appartenir à un cluster thématique clair."
                ),
            )
        with col_excl2:
            exclude_short = st.checkbox(
                "Exclure les abstracts courts",
                key="filter_exclude_short",
                help=(
                    "Exclut les articles dont l'abstract contient moins de 50 mots. "
                    "Ces articles ont souvent un résumé incomplet ou mal formaté, "
                    "ce qui peut biaiser les scores NLP (nouveauté, topics…)."
                ),
            )

    return {
        "topic_id": selected_topic,
        "lang": None if selected_lang == "Toutes" else selected_lang,
        "month_from": month_from.strip() or None,
        "month_to": month_to.strip() or None,
        "novelty_min": novelty_min if novelty_min > 0 else None,
        "search": search.strip() or None,
        "exclude_outliers": exclude_outliers,
        "exclude_short": exclude_short,
    }


def _render_table(category_id: str, filters: dict) -> Optional[str]:
    filter_key = str(sorted(filters.items()))
    if st.session_state.get("_articles_filter_key") != filter_key:
        st.session_state["_articles_filter_key"] = filter_key
        st.session_state["articles_page"] = 0
        st.session_state.pop("selected_arxiv_id", None)

    page = st.session_state.get("articles_page", 0)
    offset = page * _PAGE_SIZE
    total = get_articles_count(category_id, **filters)
    n_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)

    col_info, col_prev, col_page, col_next = st.columns([4, 1, 1, 1])
    col_info.caption(
        f"**{total:,}** articles".replace(",", "\u202f")
        + f"  ·  page {page + 1} / {n_pages}"
    )
    with col_prev:
        if st.button("◀ Préc.", disabled=(page == 0), use_container_width=True):
            st.session_state["articles_page"] = page - 1
            st.session_state.pop("selected_arxiv_id", None)
            st.rerun()
    with col_page:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px; color:#888;'>"
            f"{page + 1} / {n_pages}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button(
            "Suiv. ▶", disabled=(page >= n_pages - 1), use_container_width=True
        ):
            st.session_state["articles_page"] = page + 1
            st.session_state.pop("selected_arxiv_id", None)
            st.rerun()

    if total == 0:
        st.info("Aucun article ne correspond aux filtres sélectionnés.", icon="ℹ️")
        return None

    articles = get_articles_page(
        category_id, offset=offset, limit=_PAGE_SIZE, **filters
    )
    if not articles:
        return None

    df = pd.DataFrame(articles)

    df["arxiv_url"] = "https://arxiv.org/abs/" + df["arxiv_id"]
    df["top_keywords"] = df["keywords"].apply(_format_keywords)
    df["top_entities"] = df["entity_list"].apply(_format_entities)

    df = df.rename(
        columns={
            "arxiv_id": "ID",
            "title": "Titre",
            "first_author": "Auteur",
            "authors_count": "Auteurs",
            "published_date": "Date pub.",
            "updated": "Mis à jour",
            "primary_category": "Cat. primaire",
            "topic_label": "Topic",
            "topic_prob": "Prob. topic",
            "is_topic_outlier": "Outlier",
            "lang": "Lang",
            "novelty_score": "Nouveauté",
            "intra_similarity": "Sim. intra",
            "centroid_distance": "Dist. centroïde",
            "flesch_kincaid_grade": "Lisibilité",
            "citation_count": "Citations",
            "latex_density": "LaTeX %",
            "top_keywords": "Mots-clés",
            "top_entities": "Entités",
            "arxiv_url": "arXiv",
            "pdf_url": "PDF",
        }
    )

    display_cols = [
        "Titre",
        "Auteur",
        "Date pub.",
        "Topic",
        "Nouveauté",
        "Citations",
        "Mots-clés",
        "Entités",
        "arXiv",
        "PDF",
        "ID",
        "Auteurs",
        "Mis à jour",
        "Cat. primaire",
        "Prob. topic",
        "Outlier",
        "Lang",
        "Sim. intra",
        "Dist. centroïde",
        "Lisibilité",
        "LaTeX %",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    event = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        height=min(700, len(df) * 35 + 40),
        selection_mode="single-row",
        on_select="rerun",
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Titre": st.column_config.TextColumn("Titre", width="large"),
            "Auteur": st.column_config.TextColumn("Auteur", width="medium"),
            "Auteurs": st.column_config.NumberColumn(
                "Auteurs", format="%d", width="small"
            ),
            "Date pub.": st.column_config.DateColumn(
                "Date pub.", width="small", format="YYYY-MM-DD"
            ),
            "Mis à jour": st.column_config.DateColumn(
                "Mis à jour", width="small", format="YYYY-MM-DD"
            ),
            "Cat. primaire": st.column_config.TextColumn(
                "Cat. primaire", width="small"
            ),
            "Topic": st.column_config.TextColumn("Topic", width="medium"),
            "Prob. topic": st.column_config.NumberColumn(
                "Prob. topic", format="%.3f", width="small"
            ),
            "Outlier": st.column_config.CheckboxColumn("Outlier", width="small"),
            "Lang": st.column_config.TextColumn("Lang", width="small"),
            "Nouveauté": st.column_config.NumberColumn(
                "Nouveauté", format="%.4f", width="small"
            ),
            "Sim. intra": st.column_config.NumberColumn(
                "Sim. intra", format="%.4f", width="small"
            ),
            "Dist. centroïde": st.column_config.NumberColumn(
                "Dist. centroïde", format="%.4f", width="small"
            ),
            "Lisibilité": st.column_config.NumberColumn(
                "Lisibilité", format="%.1f", width="small"
            ),
            "Citations": st.column_config.NumberColumn(
                "Citations", format="%d", width="small"
            ),
            "Mots-clés": st.column_config.TextColumn("Mots-clés", width="large"),
            "Entités": st.column_config.TextColumn("Entités", width="large"),
            "arXiv": st.column_config.LinkColumn(
                "arXiv", display_text="Ouvrir ↗", width="small"
            ),
            "PDF": st.column_config.LinkColumn(
                "PDF", display_text="Télécharger ↗", width="small"
            ),
        },
        key="articles_table",
    )

    selection = event.selection if hasattr(event, "selection") else {}
    rows_sel = (selection.get("rows") or []) if selection else []
    if rows_sel:
        idx = rows_sel[0]
        selected_id = articles[idx]["arxiv_id"]
        st.session_state["selected_arxiv_id"] = selected_id
        return selected_id

    return st.session_state.get("selected_arxiv_id")


def _render_article_detail(detail: dict) -> None:
    st.subheader(detail["title"] or detail["arxiv_id"])

    meta_parts = []
    if detail["first_author"]:
        meta_parts.append(f"**{detail['first_author']}**")
    if detail["published_date"]:
        meta_parts.append(str(detail["published_date"]))
    if detail.get("updated") and detail["updated"] != detail["published_date"]:
        meta_parts.append(f"mis à jour le {detail['updated']}")
    if detail["topic_label"]:
        meta_parts.append(f"Topic : {detail['topic_label']}")
    if detail.get("primary_category"):
        meta_parts.append(f"Cat. : `{detail['primary_category']}`")
    if detail["lang"]:
        meta_parts.append(f"Langue : `{detail['lang']}`")
    if detail.get("is_topic_outlier"):
        meta_parts.append("⚠ outlier BERTopic")
    if detail.get("is_short_abstract"):
        meta_parts.append("⚠ abstract court")
    st.caption("  ·  ".join(meta_parts))

    link_cols = st.columns(3)
    with link_cols[0]:
        if detail.get("pdf_url"):
            st.link_button("Voir le PDF ↗", detail["pdf_url"])
        else:
            st.caption(f"arXiv ID : `{detail['arxiv_id']}`")
    with link_cols[1]:
        if detail.get("semantic_scholar_url"):
            st.link_button("Semantic Scholar ↗", detail["semantic_scholar_url"])
    with link_cols[2]:
        if detail.get("doi"):
            st.link_button(f"DOI ↗", f"https://doi.org/{detail['doi']}")

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric(
        "Nouveauté",
        (
            f"{detail['novelty_score']:.4f}"
            if detail["novelty_score"] is not None
            else "—"
        ),
        help=(
            "Score d'isolement sémantique (0→1). "
            "0 = très similaire aux articles existants du topic, "
            "1 = très différent des articles précédents. "
            "Basé sur la distance cosinus dans l'espace d'embeddings."
        ),
    )
    mc2.metric(
        "Similarité intra",
        (
            f"{detail['intra_similarity']:.4f}"
            if detail["intra_similarity"] is not None
            else "—"
        ),
        help=(
            "Similarité cosinus moyenne avec les autres articles du même topic (0→1). "
            "Proche de 1 = article typique du topic · Proche de 0 = article atypique."
        ),
    )
    mc3.metric(
        "Dist. centroïde",
        (
            f"{detail['centroid_distance']:.4f}"
            if detail.get("centroid_distance") is not None
            else "—"
        ),
        help=(
            "Distance euclidienne au centroïde sémantique du topic. "
            "Faible = article très représentatif · Élevée = article en périphérie du topic."
        ),
    )
    mc4.metric(
        "Citations",
        (
            str(detail["citation_count"])
            if detail.get("citation_count") is not None
            else "—"
        ),
        help="Nombre de citations reçues selon Semantic Scholar.",
    )
    mc5.metric(
        "Références",
        (
            str(detail["reference_count"])
            if detail.get("reference_count") is not None
            else "—"
        ),
        help="Nombre de références bibliographiques listées dans le papier.",
    )

    mc7, mc8, mc9, _mc10, _mc11, _mc12 = st.columns(6)
    mc7.metric(
        "Lisibilité (FK grade)",
        (
            f"{detail['flesch_kincaid_grade']:.1f}"
            if detail.get("flesch_kincaid_grade") is not None
            else "—"
        ),
        help=(
            "Score Flesch-Kincaid Grade Level (niveau scolaire estimé en années d'étude). "
            "Ex : 12 = niveau lycée · 16 = niveau master · 20+ = très technique. "
            "Valeur élevée = texte plus difficile à lire."
        ),
    )
    mc8.metric(
        "Densité LaTeX",
        (
            f"{detail['latex_density']:.3f}"
            if detail.get("latex_density") is not None
            else "—"
        ),
        help=(
            "Proportion de tokens LaTeX dans l'abstract (0→1). "
            "Ex : 0.15 = 15 % des tokens sont des formules ou commandes LaTeX. "
            "Élevée = article très mathématique / formel."
        ),
    )
    mc9.metric(
        "Prob. topic",
        f"{detail['topic_prob']:.3f}" if detail.get("topic_prob") is not None else "—",
        help=(
            "Probabilité d'appartenance au topic BERTopic assigné (0→1). "
            "Proche de 1 = article bien représentatif du topic · "
            "Proche de 0 = article ambiguë (potentiellement outlier)."
        ),
    )

    bib_parts = []
    if detail.get("journal_ref"):
        bib_parts.append(f"**Journal :** {detail['journal_ref']}")
    if detail.get("doi"):
        bib_parts.append(f"**DOI :** `{detail['doi']}`")
    if detail.get("lang_confidence") is not None:
        bib_parts.append(f"**Confiance langue :** {detail['lang_confidence']:.3f}")
    if detail.get("authors_count") is not None:
        bib_parts.append(f"**Nb. auteurs :** {detail['authors_count']}")
    if bib_parts:
        st.caption("  ·  ".join(bib_parts))

    if detail["abstract"]:
        with st.expander("Résumé (abstract)", expanded=True):
            st.markdown(detail["abstract"])

    col_kw, col_ner = st.columns(2)

    with col_kw:
        keywords = _parse_json_field(detail.get("keywords"))
        if keywords:
            st.markdown("**Mots-clés extraits**")
            kw_html = " ".join(
                f"<span style='background:#3B82F620; color:#3B82F6; border:1px solid #3B82F640; "
                f"border-radius:12px; padding:2px 10px; font-size:0.82em; "
                f"margin:2px; display:inline-block;'>"
                f"{_kw_label(kw)}</span>"
                for kw in keywords[:20]
            )
            st.markdown(kw_html, unsafe_allow_html=True)
        else:
            st.caption("Aucun mot-clé disponible.")

    with col_ner:
        entities = _parse_json_field(detail.get("entity_list"))
        if entities:
            _render_ner_entities(entities)
        else:
            st.caption("Aucune entité NER disponible.")


def _render_ner_entities(entities: list[dict]) -> None:
    st.markdown("**Entités détectées (NER)**")

    groups: dict[str, list[dict]] = {}
    for ent in entities:
        label = ent.get("label", "AUTRE")
        groups.setdefault(label, []).append(ent)

    priority = list(_NER_LABELS.keys())
    sorted_groups = sorted(
        groups.items(),
        key=lambda x: (priority.index(x[0]) if x[0] in priority else 999, x[0]),
    )

    for label, ents in sorted_groups:
        group_name = _NER_LABELS.get(label, label)
        ents_sorted = sorted(ents, key=lambda e: e.get("count", 1), reverse=True)

        def _count_badge(e: dict) -> str:
            if e.get("count", 1) > 1:
                return f"  <small style='opacity:.6'>×{e['count']}</small>"
            return ""

        tags_html = " ".join(
            f"<span style='background:#8B5CF620; color:#8B5CF6; border:1px solid #8B5CF640; "
            f"border-radius:12px; padding:2px 8px; font-size:0.8em; "
            f"margin:1px; display:inline-block;'>"
            f"{e['text']}{_count_badge(e)}"
            f"</span>"
            for e in ents_sorted[:10]
        )
        st.markdown(
            f"<div style='margin-bottom:6px;'>"
            f"<span style='font-size:0.75em; font-weight:600; color:#888; "
            f"text-transform:uppercase; letter-spacing:0.05em;'>{group_name}</span><br>"
            f"{tags_html}</div>",
            unsafe_allow_html=True,
        )


def _parse_json_field(value) -> list | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value if value else None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if parsed else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _kw_label(kw: dict) -> str:
    text = kw.get("keyword") or kw.get("word") or str(kw)
    score = kw.get("score")
    if score is not None:
        return f"{text} {score:.2f}"
    return text


def _format_keywords(value, top_n: int = 5) -> str:
    kws = _parse_json_field(value)
    if not kws:
        return ""
    kws_sorted = sorted(kws, key=lambda k: k.get("score") or 0, reverse=True)
    parts = []
    for kw in kws_sorted[:top_n]:
        text = kw.get("keyword") or kw.get("word") or ""
        score = kw.get("score")
        if text:
            parts.append(f"{text} {score:.2f}" if score is not None else text)
    return " · ".join(parts)


def _format_entities(value, top_n: int = 5) -> str:
    ents = _parse_json_field(value)
    if not ents:
        return ""
    ents_sorted = sorted(ents, key=lambda e: e.get("count", 1), reverse=True)
    seen: set[str] = set()
    parts: list[str] = []
    for e in ents_sorted:
        text = e.get("text", "").strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
        if len(parts) >= top_n:
            break
    return " · ".join(parts)
