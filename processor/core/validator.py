"""Validation du schéma des articles avant écriture en zone CLEAN."""

from __future__ import annotations

import pandas as pd

# Champs texte obligatoires : null, NaN ou chaîne vide alors article rejeté.
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "arxiv_id",
        "title",
        "abstract",
    }
)

# Champs liste obligatoires : la valeur doit être une liste.
LIST_COLUMNS: frozenset[str] = frozenset({"authors", "categories"})

# Colonne ajoutée aux lignes rejetées pour expliquer la raison du rejet.
DROP_REASON_COL: str = "_drop_reason"


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare un DataFrame arXiv RAW en articles valides et invalides.

    Règles appliquées dans l'ordre :
      1. Si une colonne de REQUIRED_COLUMNS est totalement absente du DataFrame,
         toutes les lignes sont rejetées.
      2. Une ligne est invalide si un champ requis est None / NaN / vide.
      3. Une ligne est invalide si 'authors' ou 'categories' n'est pas une liste.
    """
    if df.empty:
        return df.copy(), pd.DataFrame(columns=list(df.columns) + [DROP_REASON_COL])

    # Cas dégénéré : colonnes requises totalement absentes du fichier RAW
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        dropped = df.copy()
        dropped[DROP_REASON_COL] = (
            f"colonnes requises absentes : {sorted(missing_cols)}"
        )
        return pd.DataFrame(columns=df.columns), dropped.reset_index(drop=True)

    invalid_mask, reasons = _build_invalid_mask(df)

    valid_df = df[~invalid_mask].copy().reset_index(drop=True)
    dropped_df = df[invalid_mask].copy()
    dropped_df[DROP_REASON_COL] = reasons[invalid_mask].values
    dropped_df = dropped_df.reset_index(drop=True)

    return valid_df, dropped_df


def _build_invalid_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Construit le masque booléen des lignes invalides et la série des raisons.
    """
    invalid_mask = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype=str)

    # Règle 1 : champs texte requis vides ou null
    for col in sorted(REQUIRED_COLUMNS):
        col_invalid = df[col].isna() | df[col].astype(str).str.strip().eq("")
        _record_reason(invalid_mask, reasons, col_invalid, f"'{col}' vide ou None")

    # Règle 2 : champs liste requis — doit être une liste Python non-null
    for col in sorted(LIST_COLUMNS):
        if col not in df.columns:
            continue
        col_invalid = df[col].apply(lambda v: not isinstance(v, list))
        _record_reason(invalid_mask, reasons, col_invalid, f"'{col}' None ou non-liste")

    return invalid_mask, reasons


def _record_reason(
    invalid_mask: pd.Series,
    reasons: pd.Series,
    col_invalid: pd.Series,
    reason: str,
) -> None:
    """
    Marque les nouvelles lignes invalides et enregistre leur raison.
    """
    newly_invalid = col_invalid & ~invalid_mask
    reasons[newly_invalid] = reason
    invalid_mask |= col_invalid
