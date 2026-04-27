"""
Enricher : module principal de l'enrichissement des métadonnées d'articles ArXiv.
"""

from enricher.pipeline.enricher import ArxivEnricher
from enricher.config.settings import EnricherConfig

__all__ = ["ArxivEnricher", "EnricherConfig"]
