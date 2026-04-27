"""
Orchestrateur principal de la collecte arXiv.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterator

import arxiv

from ..config.constants import (
    ARXIV_MAX_BATCH_SIZE,
    ARXIV_MIN_DELAY_SEC,
    ARXIV_PAGINATION_WALL,
    BACKOFF_INITIAL_DELAY,
    BACKOFF_MAX_DELAY,
    BACKOFF_MAX_RETRIES,
    BACKOFF_MULTIPLIER,
    EARLY_STOP_THRESHOLD,
    LOG_PROGRESS_EVERY,
    _ERROR_LABELS,
)

from ..config.settings import CollectorConfig
from ..monitoring import RunHistoryMonitor
from utils.run_id import build_run_id
from ..storage import RawStorage, BaseStorage
from ..utils.date import is_within_window, resolve_date_window
from utils.logger import setup_logger
from ..utils.query import build_query
from ..monitoring.report import RunReport
from .citations import SemanticScholarClient
from .serializer import serialize_article

# Mappings arXiv
_SORT_CRITERION: dict[str, arxiv.SortCriterion] = {
    "submittedDate": arxiv.SortCriterion.SubmittedDate,
    "relevance": arxiv.SortCriterion.Relevance,
    "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
}

_SORT_ORDER: dict[str, arxiv.SortOrder] = {
    "descending": arxiv.SortOrder.Descending,
    "ascending": arxiv.SortOrder.Ascending,
}


# Collecteur principal
class ArxivCollector:
    """
    Collecteur arXiv : point d'entrée unique pour un run de collecte.

    Chaque instance représente un run configuré. La méthode run() exécute
    le run complet et retourne un RunReport, même en cas d'erreur.
    """

    def __init__(
        self,
        config: CollectorConfig,
        logger: logging.Logger | None = None,
        storage: BaseStorage | None = None,
        run_id: str | None = None,
    ) -> None:
        """
        Initialise le collecteur avec sa configuration et ses dépendances.

        Args:
            config  : Configuration complète du run (CollectorConfig).
            logger  : Logger applicatif. Créé automatiquement si absent.
            storage : Backend de stockage RAW. Filesystem local par défaut.
            run_id  : Identifiant unique du run. Généré automatiquement si absent.
        """
        self.config = config

        # Run ID : généré en premier pour pouvoir nommer le fichier de log
        # Construire la "clé" pour le run_id
        if self.config.category and self.config.query:
            key = f"{self.config.category}_{self.config.query}"
        else:
            key = self.config.category or self.config.query

        self.run_id = run_id or build_run_id(key)

        # Logger : injection ou création avec run_id pour lier log & rapport & historique
        self.logger = logger or setup_logger(
            name=f"arxiv.collector_{self.run_id}",
            log_dir=config.log_dir,
            run_id=self.run_id,
        )

        # Storage : injection ou création avec les paramètres de config
        self.storage = storage or RawStorage(
            base_path=config.raw_data_path,
            overwrite=config.overwrite_existing,
            logger=self.logger,
        )

        # Monitor d'historique
        self.monitor = RunHistoryMonitor(
            logger=self.logger,
        )

        # Client arXiv
        self._client = self._build_client()

    # Point d'entrée public
    def run(self) -> RunReport:
        """
        Exécute un run complet de collecte arXiv.
        Ne lève jamais d'exception : toute erreur est loggée et retournée dans RunReport.
        """
        run_start = datetime.now(timezone.utc)
        config_snapshot = self.config.to_dict()

        self.logger.info("DÉMARRAGE COLLECTEUR arXiv")
        self.logger.info(f"  Run ID : {self.run_id}")
        self.logger.info(f"  {self.config.summary()}")

        articles: list[dict] = []
        output_path = None
        error_message: str | None = None

        try:
            # Étape 1 : Fenêtre temporelle
            start_dt, end_dt = resolve_date_window(self.config)
            self.logger.info(
                f"Fenêtre temporelle : {start_dt.date()} to {end_dt.date()}"
            )

            # Étape 2 : Construction de la query
            query = build_query(self.config, start_dt, end_dt)
            self.logger.info(f"Query arXiv : {query}")

            # Étape 3 : Fetch paginé avec filtrage temporel
            articles = self._fetch(query, start_dt, end_dt)

            # Étape 3b : Enrichissement citations Semantic Scholar (optionnel)
            if articles and self.config.fetch_citations:
                self._enrich_citations(articles)

            # Étape 4 : Stockage RAW (uniquement si des articles ont été collectés)
            if articles:
                output_path = self.storage.save(
                    articles=articles,
                    category=self.config.category,
                    run_date=run_start,
                )

        except Exception as exc:
            self.logger.error(f"Échec du run : {exc}", exc_info=True)
            error_message = str(exc)

        # Étapes 5 & 6 : Rapport + historique
        report = RunReport.build(
            run_start=run_start,
            articles=articles,
            output_path=output_path,
            config_snapshot=config_snapshot,
            run_id=self.run_id,
            error_message=error_message,
        )

        # Warning si aucun article collecté (et pas d'erreur)
        if not articles and error_message is None:
            self.logger.warning(
                "ATTENTION : Aucun article collecté. "
                "ESSAYEZ : Élargissez la fenêtre ou simplifiez la query."
            )

        self._log_report(report)
        self.monitor.record(report, config_snapshot)

        return report

    # Fetch paginé
    def _fetch(
        self,
        query: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[dict]:
        """
        Itère sur les résultats arXiv, filtre par fenêtre temporelle
        et retourne la liste des articles sérialisés.
        """
        fetch_start = time.time()

        # Construction de l'objet de recherche arXiv
        search = arxiv.Search(
            query=query,
            max_results=self.config.max_results,
            sort_by=_SORT_CRITERION.get(
                self.config.sort_by,
                arxiv.SortCriterion.SubmittedDate,  # fallback sûr
            ),
            sort_order=_SORT_ORDER.get(
                self.config.sort_order,
                arxiv.SortOrder.Descending,  # fallback sûr
            ),
        )

        # L'arrêt anticipé n'est valide que pour submittedDate descendant
        _early_stop_enabled = (
            self.config.sort_by == "submittedDate"
            and self.config.sort_order == "descending"
        )

        articles: list[dict] = []
        out_of_window: int = 0  # Compteur d'articles consécutifs hors-fenêtre
        total_tested: int = 0  # Nombre total d'articles testés (avant filtrage)

        self.logger.info("Début de l'itération sur les résultats arXiv...")

        try:
            for result in self._fetch_with_backoff(search):
                total_tested += 1
                pub = result.published

                # Filtrage temporel : l'article est-il dans la fenêtre ?
                if not is_within_window(pub, start_dt, end_dt):
                    if _early_stop_enabled:
                        out_of_window += 1
                        if out_of_window >= EARLY_STOP_THRESHOLD:
                            self.logger.info(
                                f"Arrêt anticipé : {EARLY_STOP_THRESHOLD} articles consécutifs hors-fenêtre."
                            )
                            break
                    continue

                # Article dans la fenêtre alors réinitialisation du compteur et sérialisation
                out_of_window = 0
                articles.append(serialize_article(result))

                # Log de progression
                if len(articles) % LOG_PROGRESS_EVERY == 0:
                    elapsed = time.time() - fetch_start
                    rate = len(articles) / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"  {len(articles)} articles collectés | {total_tested} testés | {rate:.1f} art/s"
                    )

        except arxiv.HTTPError as exc:
            if exc.status == 429:
                self.logger.error(
                    f"HTTP 429 FATAL : Rate limit arXiv atteint après {BACKOFF_MAX_RETRIES} tentatives."
                )
            else:
                self.logger.error(f"HTTP {exc.status} : {exc}")

        except Exception as exc:
            self.logger.error(
                f"Erreur inattendue pendant le fetch ({type(exc).__name__}): {exc}",
                exc_info=True,
            )

        # Bilan du fetch
        elapsed = time.time() - fetch_start
        rejection_rate = (
            (total_tested - len(articles)) / total_tested * 100
            if total_tested > 0
            else 0
        )

        if not articles:
            self.logger.warning(
                f"ZÉRO article collecté : {total_tested} testés, 100% rejetés."
            )
        else:
            self.logger.info(
                f"Fetch terminé : {len(articles)} articles collectés | {total_tested} testés | "
                f"{rejection_rate:.1f}% rejetés | {elapsed:.1f}s"
            )

        return articles

    def _fetch_with_backoff(self, search: arxiv.Search) -> Iterator[arxiv.Result]:
        """
        Itérateur sur les résultats arXiv avec backoff exponentiel et reprise à l'offset.
        """
        attempt: int = 0
        delay: int = BACKOFF_INITIAL_DELAY
        offset: int = 0  # Curseur de position dans la pagination arXiv

        while True:
            try:
                for result in self._client.results(search, offset=offset):
                    yield result
                    offset += 1  # Avance le curseur après chaque article reçu
                return

            except arxiv.HTTPError as exc:
                if exc.status not in _ERROR_LABELS:
                    # HTTP 500 à haut offset = mur de pagination arXiv (~10 000 résultats max)
                    if exc.status == 500 and offset >= ARXIV_PAGINATION_WALL:
                        self.logger.warning(
                            "Limite de pagination arXiv atteinte à l'offset %d. "
                            "L'API arXiv refuse les requêtes au-delà de ~10 000 résultats. "
                            "Pour collecter plus d'articles, découpez la fenêtre temporelle en mois.",
                            offset,
                        )
                        return
                    raise

                label = _ERROR_LABELS.get(exc.status, f"HTTP {exc.status}")

                # Abandon définitif
                if attempt >= BACKOFF_MAX_RETRIES:
                    self.logger.error(
                        f"{label} FATAL : {BACKOFF_MAX_RETRIES} tentatives épuisées (offset={offset})."
                    )
                    raise

                # Calcul du délai d'attente (plafonné à BACKOFF_MAX_DELAY)
                wait = min(delay, BACKOFF_MAX_DELAY)
                self.logger.warning(
                    f"{label} : Backoff {wait}s (tentative {attempt+1}/{BACKOFF_MAX_RETRIES}, offset={offset})..."
                )
                time.sleep(wait)

                attempt += 1
                delay = min(delay * BACKOFF_MULTIPLIER, BACKOFF_MAX_DELAY)

    # Client arXiv
    def _build_client(self) -> arxiv.Client:
        """
        Instancie le client arXiv avec les paramètres.
        """
        # Forçage du délai minimum pour éviter les HTTP 429
        safe_delay = max(self.config.delay_between_batches, ARXIV_MIN_DELAY_SEC)

        if safe_delay > self.config.delay_between_batches:
            self.logger.warning(
                f"Délai inter-requêtes ajusté : "
                f"{self.config.delay_between_batches}s to {safe_delay}s"
            )

        return arxiv.Client(
            page_size=min(self.config.batch_size, ARXIV_MAX_BATCH_SIZE),
            delay_seconds=safe_delay,
            num_retries=1,
        )

    def _enrich_citations(self, articles: list[dict]) -> None:
        """
        Enrichit chaque article avec citation_count depuis Semantic Scholar.
        """
        self.logger.info(
            f"Enrichissement citations Semantic Scholar : {len(articles)} articles..."
        )
        client = SemanticScholarClient(
            batch_size=self.config.citations_batch_size,
            api_key=self.config.citations_api_key,
            logger=self.logger,
        )
        client.enrich(articles)

    def _log_report(self, report: RunReport) -> None:
        """
        Affiche le rapport de fin de run dans les logs.
        """
        for line in report.summary_lines():
            self.logger.info(line)
