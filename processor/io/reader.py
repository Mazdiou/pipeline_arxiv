"""
Lecture des fichiers JSON de la zone RAW vers un pd.DataFrame.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def discover_raw_files(
    raw_path: str,
    category: str = "",
    year: int | None = None,
    month: int | None = None,
) -> list[Path]:
    """
    Découvre tous les fichiers articles.json dans la zone RAW correspondant aux filtres.
    """
    root = Path(raw_path)
    if not root.exists():
        return []

    # Niveau catégorie
    if category:
        cat_dirs = [root / category] if (root / category).is_dir() else []
    else:
        cat_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    files: list[Path] = []

    for cat_dir in cat_dirs:
        # Niveau année
        if year is not None:
            year_dirs = [cat_dir / f"{year:04d}"]
        else:
            year_dirs = sorted(d for d in cat_dir.iterdir() if d.is_dir())

        for year_dir in year_dirs:
            if not year_dir.exists():
                continue

            # Niveau mois
            if month is not None:
                month_dirs = [year_dir / f"{month:02d}"]
            else:
                month_dirs = sorted(d for d in year_dir.iterdir() if d.is_dir())

            for month_dir in month_dirs:
                if not month_dir.exists():
                    continue
                candidate = month_dir / "articles.json"
                if candidate.exists():
                    files.append(candidate)

    return sorted(files)


def read_raw_file(path: Path) -> pd.DataFrame:
    """
    Lit un fichier articles.json de la zone RAW et retourne un DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Fichier RAW introuvable : {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON malformé dans {path} : {exc}") from exc

    articles = raw.get("articles")
    if articles is None:
        raise ValueError(
            f"Clé 'articles' absente dans {path}. "
            f"Clés présentes : {list(raw.keys())}"
        )

    if not articles:
        return pd.DataFrame()

    return pd.DataFrame(articles)
