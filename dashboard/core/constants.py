"""
dashboard/core/constants.py

Constantes partagées entre les pages du dashboard.
"""

from __future__ import annotations

# Couleurs par état de tendance (utilisées sur Page 1, 2, 3…)
STATUS_COLORS: dict[str, str] = {
    "NEW":          "#06B6D4",
    "EMERGING":     "#00C49A",
    "TRENDING":     "#3B82F6",
    "TRANSFORMING": "#8B5CF6",
    "STABLE":       "#6B7280",
    "DECLINING":    "#F59E0B",
}

# Labels lisibles des états (français)
STATUS_LABELS: dict[str, str] = {
    "NEW":          "Nouveau",
    "EMERGING":     "Émergent",
    "TRENDING":     "En croissance",
    "TRANSFORMING": "En transformation",
    "STABLE":       "Stable",
    "DECLINING":    "En déclin",
}

# Ordre canonique d'affichage des états
STATUS_ORDER: list[str] = [
    "EMERGING",
    "TRENDING",
    "TRANSFORMING",
    "STABLE",
    "DECLINING",
    "NEW",
]

# Espace fine insécable — utilisé pour formater les grands nombres (ex : 1 234 articles)
NARROW_SPACE: str = "\u202f"
