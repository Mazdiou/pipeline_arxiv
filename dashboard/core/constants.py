"""
dashboard/core/constants.py

Constantes partagées entre les pages du dashboard.
"""

from __future__ import annotations

# Couleurs par état de tendance (utilisées sur Page 1, 2, 3…)
STATUS_COLORS: dict[str, str] = {
    "NEW": "#06B6D4",
    "EMERGING": "#00C49A",
    "TRENDING": "#3B82F6",
    "TRANSFORMING": "#8B5CF6",
    "STABLE": "#6B7280",
    "DECLINING": "#F59E0B",
}

# Labels lisibles des états (français)
STATUS_LABELS: dict[str, str] = {
    "NEW": "Nouveau",
    "EMERGING": "Émergent",
    "TRENDING": "En croissance",
    "TRANSFORMING": "En transformation",
    "STABLE": "Stable",
    "DECLINING": "En déclin",
}

# Descriptions courtes des états — affichées en caption sous chaque onglet
STATUS_DESCRIPTIONS: dict[str, str] = {
    "NEW": "≤ 2 mois d'activité — trop récent pour classifier",
    "EMERGING": "Faible volume mais croissance détectée et articles novateurs — sujet en pré-émergence",
    "TRENDING": "Croissance soutenue et statistiquement confirmée (Mann-Kendall + Hurst)",
    "TRANSFORMING": "Volume stable mais fort changement sémantique — le topic mute sans croître",
    "STABLE": "Présence régulière sans tendance marquée",
    "DECLINING": "Tendance baissière statistiquement confirmée",
}

# Icônes par état (cohérentes entre toutes les pages)
STATUS_ICONS: dict[str, str] = {
    "NEW": "🆕",
    "EMERGING": "🌱",
    "TRENDING": "📈",
    "TRANSFORMING": "🔄",
    "STABLE": "⚡",
    "DECLINING": "📉",
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


def fmt_number(n: int | float | None) -> str:
    """Formate un entier avec espace fine insécable comme séparateur de milliers (style FR).

    Retourne '—' si la valeur est None.
    """
    if n is None:
        return "—"
    return f"{int(n):,}".replace(",", NARROW_SPACE)
