"""
Client Semantic Scholar pour l'enrichissement des citations arXiv.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request

from ..config.constants import (
    SS_BACKOFF_INITIAL_DELAY,
    SS_BACKOFF_MAX_DELAY,
    SS_BACKOFF_MAX_RETRIES,
    SS_BATCH_ENDPOINT,
    SS_REQUEST_DELAY,
    SS_REQUEST_DELAY_NO_KEY,
    SS_TIMEOUT,
)

_SS_FIELDS = "citationCount,referenceCount,url"


class SemanticScholarClient:
    """
    Client pour récupérer citation_count et reference_count depuis Semantic Scholar.
    """

    def __init__(
        self,
        batch_size: int,
        api_key: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.batch_size = batch_size
        self._api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        # Délai inter-lots adapté à la présence ou non d'une clé API
        self._request_delay = SS_REQUEST_DELAY if api_key else SS_REQUEST_DELAY_NO_KEY

    def enrich(self, articles: list[dict]) -> None:
        """
        Ajoute citation_count et reference_count à chaque article.
        """
        total = len(articles)
        enriched = 0

        for i in range(0, total, self.batch_size):
            batch = articles[i : i + self.batch_size]
            results = self._fetch_batch([a["arxiv_id_short"] for a in batch])

            for article, ss_data in zip(batch, results):
                if ss_data is not None:
                    article["citation_count"] = ss_data.get("citationCount")
                    article["reference_count"] = ss_data.get("referenceCount")
                    article["semantic_scholar_url"] = ss_data.get("url")
                    enriched += 1

            # Respect du rate limit entre les lots (avec jitter ±20%)
            if i + self.batch_size < total:
                jitter = self._request_delay * random.uniform(-0.2, 0.2)
                time.sleep(max(0.0, self._request_delay + jitter))

        self.logger.info(
            f"Citations Semantic Scholar : {enriched}/{total} articles enrichis."
        )

    def _fetch_batch(self, arxiv_ids: list[str]) -> list[dict | None]:
        """
        Appelle le batch endpoint avec backoff sur 429.
        """
        ids = [f"arXiv:{aid}" for aid in arxiv_ids]
        payload = json.dumps({"ids": ids}).encode()

        req = urllib.request.Request(
            url=f"{SS_BATCH_ENDPOINT}?fields={_SS_FIELDS}",
            data=payload,
            headers=self._build_headers(),
            method="POST",
        )

        attempt = 0
        delay = SS_BACKOFF_INITIAL_DELAY

        while True:
            try:
                with urllib.request.urlopen(req, timeout=SS_TIMEOUT) as resp:
                    return json.loads(resp.read())

            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < SS_BACKOFF_MAX_RETRIES:
                    # Respecter Retry-After si présent, sinon backoff exponentiel
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = int(retry_after)
                        except ValueError:
                            wait = delay
                    else:
                        wait = min(delay, SS_BACKOFF_MAX_DELAY)
                    self.logger.warning(
                        f"Semantic Scholar HTTP 429 : attente {wait}s "
                        f"(tentative {attempt + 1}/{SS_BACKOFF_MAX_RETRIES})..."
                    )
                    time.sleep(wait)
                    attempt += 1
                    delay = min(delay * 2, SS_BACKOFF_MAX_DELAY)
                else:
                    self.logger.warning(
                        f"Semantic Scholar HTTP {exc.code} : lot de {len(arxiv_ids)} articles ignoré."
                    )
                    return [None] * len(arxiv_ids)

            except Exception as exc:
                self.logger.warning(
                    f"Semantic Scholar indisponible ({type(exc).__name__}) : "
                    f"lot de {len(arxiv_ids)} articles ignoré."
                )
                return [None] * len(arxiv_ids)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers
