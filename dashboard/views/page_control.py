from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

import streamlit as st

from dashboard.core.config_store import (
    CategoryConfig,
    ConfigStore,
    normalize_category_id,
    validate_arxiv_id,
)
from dashboard.core.constants import fmt_number
from dashboard.core.db import (
    CategoryRow,
    delete_category_all,
    get_all_category_rows,
    get_category_counts,
    get_global_stats,
)
from dashboard.core.airflow_client import (
    is_airflow_available,
    trigger_bootstrap,
    trigger_reenrich,
    get_recent_runs,
    get_running_bootstraps,
)

logger = logging.getLogger(__name__)

_store = ConfigStore()

TAXONOMY_URL = "https://arxiv.org/category_taxonomy"

DATE_MODE_LOOKBACK = "Lookback (nombre de jours)"
DATE_MODE_PERIOD = "Période fixe (start → end)"
DATE_MODES = [DATE_MODE_LOOKBACK, DATE_MODE_PERIOD]

SORT_BY_OPTIONS = {
    "submittedDate": "Date de soumission",
    "lastUpdatedDate": "Dernière mise à jour",
    "relevance": "Pertinence (query)",
}
SORT_ORDER_OPTIONS = {
    "descending": "Décroissant (récent → ancien)",
    "ascending": "Croissant (ancien → récent)",
}


def render() -> None:
    _init_session_state()
    _render_header()
    _render_kpi_bar()
    st.divider()
    _render_category_section()
    st.divider()
    _render_pipeline_status()


def _init_session_state() -> None:
    defaults = {
        "show_form": False,
        "edit_category": None,
        "confirm_delete": None,
        "pending_bootstraps": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_header() -> None:
    st.title("Contrôle du pipeline")
    st.caption("Gérez les catégories surveillées et synchronisez le dashboard.")


def _render_kpi_bar() -> None:
    stats = get_global_stats()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Catégories actives",
            value=stats["n_categories"],
        )
    with col2:
        st.metric(
            label="Articles indexés",
            value=fmt_number(stats["n_articles"]),
        )
    with col3:
        last = stats["last_sync"]
        if last is None:
            label = "—"
        else:
            try:
                if hasattr(last, "tzinfo") and last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                now = datetime.now(tz=timezone.utc)
                total_sec = int((now - last).total_seconds())
                if total_sec < 60:
                    label = "à l'instant"
                elif total_sec < 3600:
                    label = f"il y a {total_sec // 60} min"
                else:
                    label = f"il y a {total_sec // 3600}h"
            except Exception:
                label = str(last)[:16]
        st.metric(
            label="Dernière sync DuckDB",
            value=label,
            help=(
                "Date de la dernière synchronisation complète des données dans le dashboard. "
                "Mise à jour automatiquement le 1ᵉʳ de chaque mois (DAG Monthly Enrich) "
                "ou manuellement via le bouton **Sync** sur chaque card."
            ),
        )


def _render_category_section() -> None:
    db_rows = get_all_category_rows()
    configs = _store.load_all()

    db_rows_by_id = {r.category: r for r in db_rows}

    all_rows: list[CategoryRow] = []
    for cat_id, cfg in configs.items():
        if cat_id in db_rows_by_id:
            all_rows.append(db_rows_by_id[cat_id])
        else:
            all_rows.append(
                CategoryRow(
                    category=cat_id,
                    display_name=cfg.display_name,
                    bootstrapped_at=None,
                    last_synced_at=None,
                    n_articles=0,
                    n_topics=0,
                    n_months=0,
                )
            )
    all_rows.sort(key=lambda r: r.category)

    col_title, col_btn = st.columns([6, 2])
    with col_title:
        st.subheader("Catégories surveillées")
    with col_btn:
        if not st.session_state.show_form:
            if st.button("+ Ajouter une catégorie", use_container_width=True):
                st.session_state.show_form = True
                st.session_state.edit_category = None
                st.rerun()

    if not all_rows and not st.session_state.show_form:
        st.info(
            "Aucune catégorie configurée. "
            "Cliquez sur **+ Ajouter une catégorie** pour commencer.",
            icon="ℹ️",
        )

    airflow_ok = is_airflow_available()
    running_bootstraps = get_running_bootstraps() if airflow_ok else set()
    st.session_state.pending_bootstraps -= running_bootstraps
    running_bootstraps |= st.session_state.pending_bootstraps

    if all_rows:
        pairs = [all_rows[i : i + 2] for i in range(0, len(all_rows), 2)]
        for pair in pairs:
            cols = st.columns(2)
            for col, row in zip(cols, pair):
                with col:
                    _render_category_card(
                        row, configs.get(row.category), airflow_ok, running_bootstraps
                    )

    if st.session_state.show_form:
        st.divider()
        _render_form()

    if st.session_state.confirm_delete:
        _delete_dialog(st.session_state.confirm_delete)


def _render_category_card(
    row: CategoryRow,
    config: Optional[CategoryConfig],
    airflow_ok: bool,
    running_bootstraps: set[str] | None = None,
) -> None:
    status_icons = {"ok": "🟢", "warning": "🟡", "error": "🔴"}
    icon = status_icons.get(row.status, "⚪")

    with st.container(border=True):
        c_title, c_menu = st.columns([5, 1])
        with c_title:
            st.markdown(f"**{row.display_name}**")
            st.caption(f"`{row.category}`")
        with c_menu:
            airflow_disabled_help = (
                "Airflow indisponible — démarrez le serveur Airflow."
            )
            is_running = (
                running_bootstraps is not None and row.category in running_bootstraps
            )
            pipeline_busy_help = "Pipeline en cours — revenez dans environ une heure."

            menu = st.popover("⋯", use_container_width=True)
            with menu:
                if st.button(
                    "Modifier la config",
                    key=f"edit_{row.category}",
                    use_container_width=True,
                ):
                    st.session_state.show_form = True
                    st.session_state.edit_category = row.category
                    st.rerun()

                st.divider()

                if row.n_articles > 0:
                    if st.button(
                        "Re-lancer le bootstrap (recollecte complète)",
                        key=f"reboot_{row.category}",
                        use_container_width=True,
                        disabled=not airflow_ok or is_running,
                        help=(
                            pipeline_busy_help
                            if is_running
                            else (
                                (
                                    "Relance la collecte complète depuis la config. "
                                    "Utile si vous avez étendu la période, augmenté max_results "
                                    "ou si des données sont manquantes. "
                                    "⚠ Opération longue (30 min à plusieurs heures selon la catégorie)."
                                )
                                if airflow_ok
                                else airflow_disabled_help
                            )
                        ),
                    ):
                        _trigger_bootstrap(row.category)

                if st.button(
                    "Re-enrichir",
                    key=f"reenrich_{row.category}",
                    use_container_width=True,
                    disabled=not airflow_ok or is_running,
                    help=(
                        pipeline_busy_help
                        if is_running
                        else (
                            "Relance uniquement NLP + tendances sur les données existantes."
                            if airflow_ok
                            else airflow_disabled_help
                        )
                    ),
                ):
                    _trigger_reenrich(row.category)

                st.divider()

                if st.button(
                    "Supprimer",
                    key=f"del_{row.category}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.confirm_delete = row.category
                    st.rerun()

        m1, m2, m3 = st.columns(3)
        m1.metric("Articles", fmt_number(row.n_articles))
        m2.metric("Topics", row.n_topics)
        m3.metric("Mois", row.n_months)

        st.markdown(f"{icon} **{row.status_label}** · Sync : {row.last_synced_label}")

        if row.n_articles == 0:
            is_running = (
                running_bootstraps is not None and row.category in running_bootstraps
            )
            if is_running:
                st.info(
                    "Pipeline en cours — revenez dans environ une heure.",
                    icon="⏳",
                )
            else:
                if st.button(
                    "Lancer le pipeline (bootstrap initial)",
                    key=f"boot_{row.category}",
                    use_container_width=True,
                    disabled=not airflow_ok,
                    help=(
                        "Lance la collecte initiale complète via Airflow : "
                        "collecte arXiv → traitement → NLP → tendances → sync DuckDB. "
                        "⚠ Opération longue (30 min à plusieurs heures)."
                        if airflow_ok
                        else airflow_disabled_help
                    ),
                ):
                    _trigger_bootstrap(row.category)


def _trigger_bootstrap(category: str) -> None:
    with st.spinner(f"Déclenchement du bootstrap pour {category}…"):
        success, message = trigger_bootstrap(category)
    if success:
        st.toast(f"Bootstrap {category} déclenché !", icon="🚀")
        _log(f"{category} — Bootstrap lancé via Airflow.", level="info")
        st.session_state.pending_bootstraps.add(category)
        st.rerun()
    else:
        st.error(f"Impossible de lancer le bootstrap : {message}")
        _log(f"{category} — Bootstrap échoué : {message}", level="error")


def _trigger_reenrich(category: str) -> None:
    with st.spinner(f"Déclenchement du re-enrichissement pour {category}…"):
        success, message = trigger_reenrich(category)
    if success:
        st.toast(f"Re-enrichissement {category} déclenché !", icon="🔄")
        _log(f"{category} — Re-enrichissement lancé via Airflow.", level="info")
        st.rerun()
    else:
        st.error(f"Impossible de lancer le re-enrichissement : {message}")
        _log(f"{category} — Re-enrichissement échoué : {message}", level="error")


def _render_form() -> None:
    is_edit = st.session_state.edit_category is not None
    edit_cat_id = st.session_state.edit_category
    existing_cfg = _store.get(edit_cat_id) if is_edit else None

    title = f"Modifier — {edit_cat_id}" if is_edit else "Ajouter une catégorie"
    st.subheader(title)

    st.markdown("##### Identité")

    col_id, col_link = st.columns([5, 1])
    with col_id:
        arxiv_input = st.text_input(
            "Identifiant arXiv *",
            value=existing_cfg.arxiv_category if existing_cfg else "",
            placeholder="ex: stat.ML, cs.AI, math.AP",
            disabled=is_edit,
            help="Format : primary.Subcategory (ex: stat.ML).",
        )
    with col_link:
        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button("Taxonomie ↗", TAXONOMY_URL, use_container_width=True)

    display_name_input = st.text_input(
        "Nom affiché *",
        value=existing_cfg.display_name if existing_cfg else "",
        placeholder="ex: Statistics - Machine Learning",
    )

    query_input = st.text_input(
        "Mots-clés supplémentaires",
        value=existing_cfg.query if existing_cfg else "",
        placeholder="ex: large language models (optionnel)",
        help="Combinés à l'identifiant arXiv avec AND lors de la collecte.",
    )

    st.markdown("##### Collecte")

    default_date_mode = (
        DATE_MODE_PERIOD
        if (existing_cfg and existing_cfg.start_date is not None)
        else DATE_MODE_LOOKBACK
    )
    date_mode = st.radio(
        "Fenêtre temporelle",
        options=DATE_MODES,
        index=DATE_MODES.index(default_date_mode),
        horizontal=True,
        help=(
            "**Lookback** : collecte les N derniers jours depuis aujourd'hui. "
            "**Période fixe** : collecte entre deux dates précises."
        ),
    )

    lookback_days_input: int = 180
    start_date_str: Optional[str] = None
    end_date_str: Optional[str] = None

    if date_mode == DATE_MODE_LOOKBACK:
        lookback_days_input = int(
            st.number_input(
                "Nombre de jours en arrière",
                min_value=7,
                max_value=3650,
                value=existing_cfg.lookback_days if existing_cfg else 180,
                step=30,
                help="La collecte couvre les N derniers jours depuis aujourd'hui (UTC).",
            )
        )
    else:
        col_start_date, col_end_date = st.columns(2)
        with col_start_date:
            default_start = (
                date.fromisoformat(existing_cfg.start_date)
                if (existing_cfg and existing_cfg.start_date)
                else date(2024, 1, 1)
            )
            picked_start = st.date_input(
                "Date de début *",
                value=default_start,
                min_value=date(1991, 1, 1),
                max_value=date.today(),
                format="YYYY-MM-DD",
            )
            start_date_str = picked_start.isoformat()
        with col_end_date:
            has_existing_end = (
                existing_cfg.end_date is not None if existing_cfg else False
            )
            use_end_date = st.checkbox(
                "Définir une date de fin", value=has_existing_end
            )
            if use_end_date:
                default_end = (
                    date.fromisoformat(existing_cfg.end_date)
                    if (existing_cfg and existing_cfg.end_date)
                    else date.today()
                )
                picked_end = st.date_input(
                    "Date de fin",
                    value=default_end,
                    min_value=date(1991, 1, 1),
                    max_value=date.today(),
                    format="YYYY-MM-DD",
                )
                end_date_str = picked_end.isoformat()

    max_results = st.slider(
        "Nombre max d'articles à collecter (par run)",
        min_value=1000,
        max_value=9900,
        value=existing_cfg.max_results if existing_cfg else 6000,
        step=100,
        help=(
            "Limite le nombre d'articles récupérés par appel à l'API arXiv. "
            "Une valeur élevée ralentit la collecte mais améliore la couverture. "
            "Recommandé : 6 000 pour un équilibre vitesse / complétude."
        ),
    )

    fetch_citations = st.toggle(
        "Récupérer les citations",
        value=existing_cfg.fetch_citations if existing_cfg else True,
        help="Enrichit les articles via Semantic Scholar (plus lent).",
    )

    sort_by_keys = list(SORT_BY_OPTIONS.keys())
    sort_by_names = list(SORT_BY_OPTIONS.values())
    default_sort_by = existing_cfg.sort_by if existing_cfg else "submittedDate"

    sort_order_keys = list(SORT_ORDER_OPTIONS.keys())
    sort_order_names = list(SORT_ORDER_OPTIONS.values())
    default_sort_order = existing_cfg.sort_order if existing_cfg else "descending"

    col_sort_by, col_sort_order = st.columns(2)
    with col_sort_by:
        sort_by_index = st.selectbox(
            "Trier par",
            options=range(len(sort_by_keys)),
            format_func=lambda i: sort_by_names[i],
            index=(
                sort_by_keys.index(default_sort_by)
                if default_sort_by in sort_by_keys
                else 0
            ),
            help="Critère de tri des résultats de l'API arXiv.",
        )
        sort_by_val = sort_by_keys[sort_by_index]
    with col_sort_order:
        sort_order_index = st.selectbox(
            "Ordre",
            options=range(len(sort_order_keys)),
            format_func=lambda i: sort_order_names[i],
            index=(
                sort_order_keys.index(default_sort_order)
                if default_sort_order in sort_order_keys
                else 0
            ),
        )
        sort_order_val = sort_order_keys[sort_order_index]

    with st.expander("Paramètres avancés — Topics"):
        st.caption(
            "Laissez vide pour laisser BERTopic déterminer automatiquement "
            "le nombre de topics optimal."
        )
        fix_topics = st.checkbox(
            "Fixer le nombre de topics",
            value=existing_cfg.nr_topics is not None if existing_cfg else False,
        )
        nr_topics_input: Optional[int] = None
        if fix_topics:
            nr_topics_input = int(
                st.number_input(
                    "Nombre max de topics",
                    min_value=2,
                    max_value=200,
                    value=(
                        existing_cfg.nr_topics
                        if (existing_cfg and existing_cfg.nr_topics)
                        else 20
                    ),
                    step=1,
                    help="BERTopic fusionnera les topics similaires pour atteindre ce nombre.",
                )
            )

    with st.expander("Paramètres avancés — Tendances"):
        min_months = st.slider(
            "Historique minimum avant reporting (mois)",
            min_value=2,
            max_value=12,
            value=existing_cfg.min_months_for_trend if existing_cfg else 5,
            help="Nombre de mois requis pour commencer à calculer les tendances.",
        )
        z_window = st.slider(
            "Fenêtre z-score (mois)",
            min_value=2,
            max_value=12,
            value=existing_cfg.z_score_window if existing_cfg else 4,
            help="Nombre de mois comparés pour détecter une accélération.",
        )
        semantic_drift = st.toggle(
            "Dérive sémantique",
            value=existing_cfg.semantic_drift_enabled if existing_cfg else True,
            help=(
                "Active la détection de mutation sémantique (DST). "
                "Nécessite les embeddings générés par le module Enricher. "
                "Désactiver si les embeddings sont absents ou pour accélérer le traitement."
            ),
        )

    col_cancel, col_save = st.columns([1, 1])
    with col_cancel:
        if st.button("Annuler", use_container_width=True):
            _close_form()
    with col_save:
        if st.button("Enregistrer", type="primary", use_container_width=True):
            _submit_form(
                arxiv_input=arxiv_input,
                display_name_input=display_name_input,
                query_input=query_input,
                lookback_days=lookback_days_input,
                start_date=start_date_str,
                end_date=end_date_str,
                max_results=max_results,
                fetch_citations=fetch_citations,
                sort_by=sort_by_val,
                sort_order=sort_order_val,
                nr_topics=nr_topics_input,
                min_months=min_months,
                z_window=z_window,
                semantic_drift=semantic_drift,
                is_edit=is_edit,
                edit_cat_id=edit_cat_id,
            )


def _submit_form(
    arxiv_input: str,
    display_name_input: str,
    query_input: str,
    lookback_days: int,
    start_date: Optional[str],
    end_date: Optional[str],
    max_results: int,
    fetch_citations: bool,
    sort_by: str,
    sort_order: str,
    nr_topics: Optional[int],
    min_months: int,
    z_window: int,
    semantic_drift: bool,
    is_edit: bool,
    edit_cat_id: Optional[str],
) -> None:
    errors: list[str] = []

    if not is_edit:
        err = validate_arxiv_id(arxiv_input)
        if err:
            errors.append(err)
        else:
            cat_id = normalize_category_id(arxiv_input)
            if _store.exists(cat_id):
                errors.append(
                    f"La catégorie '{cat_id}' est déjà configurée. "
                    "Utilisez le menu ⋯ pour la modifier."
                )

    if not display_name_input.strip():
        errors.append("Le nom affiché est obligatoire.")

    if errors:
        for err in errors:
            st.error(err)
        return

    try:
        cat_id = edit_cat_id if is_edit else normalize_category_id(arxiv_input)
        cfg = CategoryConfig(
            display_name=display_name_input.strip(),
            arxiv_category=arxiv_input.strip(),
            query=query_input.strip(),
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            fetch_citations=fetch_citations,
            sort_by=sort_by,
            sort_order=sort_order,
            nr_topics=nr_topics,
            min_months_for_trend=min_months,
            z_score_window=z_window,
            semantic_drift_enabled=semantic_drift,
        )
        _store.upsert(cat_id, cfg)
    except ValueError as exc:
        st.error(f"Configuration invalide : {exc}")
        return
    except Exception as exc:
        st.error(f"Erreur lors de la sauvegarde : {exc}")
        logger.error("_submit_form échoué : %s", exc)
        return

    action = "modifiée" if is_edit else "ajoutée"
    st.toast(f"Catégorie {cat_id} {action}.", icon="✅")
    _log(f"{cat_id} — Configuration {action}.", level="info")
    _close_form()


def _close_form() -> None:
    st.session_state.show_form = False
    st.session_state.edit_category = None
    st.rerun()


@st.dialog("Confirmer la suppression")
def _delete_dialog(category: str) -> None:
    counts = get_category_counts(category)

    st.warning(
        f"Cette action supprimera définitivement toutes les données de **{category}**.",
        icon="⚠️",
    )

    st.markdown("**Données qui seront supprimées :**")
    c1, c2 = st.columns(2)
    c1.metric("Articles", fmt_number(counts.n_articles))
    c2.metric("Topics", counts.n_topics)
    c1.metric("Rapports tendance", counts.n_trend_reports)
    c2.metric("Entrées mensuelles", counts.n_monthly_entries)

    st.markdown("La configuration associée sera également supprimée.")
    st.markdown("---")

    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("Annuler", use_container_width=True):
            st.session_state.confirm_delete = None
            st.rerun()
    with col_confirm:
        if st.button(
            "Supprimer définitivement",
            type="primary",
            use_container_width=True,
        ):
            _do_delete(category)


def _do_delete(category: str) -> None:
    with st.spinner(f"Suppression de {category}…"):
        ok_db, msg_db = delete_category_all(category)

    if not ok_db:
        st.error(f"Erreur lors de la suppression DuckDB : {msg_db}")
        _log(f"{category} — Suppression échouée : {msg_db}", level="error")
        return

    ok_json = _store.delete(category)
    if not ok_json:
        logger.warning(
            "delete '%s' — entrée absente de pipeline_configs.json", category
        )

    _log(f"{category} — Catégorie supprimée.", level="info")
    st.session_state.confirm_delete = None
    st.toast(f"Catégorie {category} supprimée.", icon="🗑️")
    st.rerun()


@st.fragment(run_every=15)
def _render_pipeline_status() -> None:
    st.subheader("Statut des pipelines")

    if not is_airflow_available():
        st.caption(
            "Airflow indisponible — démarrez le serveur pour voir l'état des runs."
        )
        return

    runs = get_recent_runs(limit=15)

    if not runs:
        st.caption("Aucun run trouvé.")
        return

    state_icons = {
        "running": "🔄",
        "success": "✅",
        "failed": "❌",
        "queued": "⏳",
        "skipped": "⏭️",
    }
    dag_labels = {
        "arxiv_bootstrap": "Bootstrap",
        "arxiv_daily_watch": "Daily Watch",
    }

    for run in runs:
        state = run["state"]
        icon = state_icons.get(state, "⚪")
        category = run["category"]
        dag_label = dag_labels.get(run["dag_id"], run["dag_id"])

        duration_str = ""
        if run["start_date"]:
            try:
                start = datetime.fromisoformat(run["start_date"].replace("Z", "+00:00"))
                if run["end_date"]:
                    end = datetime.fromisoformat(run["end_date"].replace("Z", "+00:00"))
                else:
                    end = datetime.now(tz=timezone.utc)
                secs = int((end - start).total_seconds())
                if secs < 60:
                    duration_str = f"{secs}s"
                elif secs < 3600:
                    duration_str = f"{secs // 60}m {secs % 60}s"
                else:
                    duration_str = f"{secs // 3600}h {(secs % 3600) // 60}m"
            except Exception:
                pass

        start_label = ""
        if run["start_date"]:
            try:
                start_label = run["start_date"][:16].replace("T", " ")
            except Exception:
                pass

        parts = [f"{icon} **{category}**", f"`{dag_label}`", f"— {state}"]
        if duration_str:
            parts.append(f"· {duration_str}")
        if start_label:
            parts.append(f"· {start_label} UTC")

        st.markdown("  ".join(parts))


def _log(message: str, level: str = "info") -> None:
    if level == "error":
        logger.error(message)
    else:
        logger.info(message)
