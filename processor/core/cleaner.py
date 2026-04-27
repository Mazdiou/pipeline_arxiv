"""
Nettoyage des champs textuels (title, abstract, comment) des articles arXiv.
"""

from __future__ import annotations

import html
import re
import unicodedata

MATH_PLACEHOLDER: str = "[MATH]"
"""
Remplace les expressions mathématiques supprimées.
"""

# Patterns LaTeX compilés

# 1. Environnements LaTeX : \begin{equation}...\end{equation}
_RE_ENV = re.compile(
    r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}",
    re.DOTALL,
)

# 2. Math display : $$...$$  (avant inline pour éviter la confusion)
_RE_MATH_DISPLAY = re.compile(r"\$\$[^$]*?\$\$", re.DOTALL)

# 3. Math inline : $...$
_RE_MATH_INLINE = re.compile(r"\$[^$\n]+?\$")

# 4. Commandes dont on conserve le contenu : \textbf{hello} = hello
_RE_CMD_KEEP = re.compile(
    r"\\(?:textbf|textit|textrm|texttt|emph|text|"
    r"mathrm|mathbf|mathit|mathcal|mathbb|mathsf|mathtt|"
    r"operatorname|boldsymbol|"
    r"underline|overline|widehat|widetilde|hat|tilde|bar|vec|"
    r"sqrt|frac)\{([^}]*)\}"
)

# 5. Commandes à supprimer entièrement (y compris leur argument)
_RE_CMD_DROP = re.compile(
    r"\\(?:cite|citep|citet|citealt|nocite|"
    r"ref|eqref|pageref|label|"
    r"footnote|footnotemark|footnotetext|"
    r"url|href|hyperref|hyperlink|"
    r"index|glossary)\{[^}]*\}"
)

# 6. Commandes simples sans argument (\noindent, \par, \newline, \alpha, etc.)
_RE_CMD_SIMPLE = re.compile(r"\\[a-zA-Z]+\b\s*")

# 7. Accolades orphelines restantes après les substitutions précédentes
_RE_BRACES = re.compile(r"[{}]")

# Caractères Unicode invisibles
_RE_INVISIBLE_SUPPRESS = re.compile(r"[\u00AD\u200B\u200C\u200D\uFEFF]")

# Espaces et tabulations multiples alors espace unique
_RE_SPACES = re.compile(r"[ \t]+")

# Lignes vides multiples alors simple saut de ligne
_RE_NEWLINES = re.compile(r"\n{2,}")


# Fonctions publiques
def clean_latex(text: str) -> str:
    """
    Supprime les marqueurs LaTeX d'une chaîne de texte.
    """
    text = _RE_ENV.sub("", text)
    text = _RE_MATH_DISPLAY.sub(MATH_PLACEHOLDER, text)
    text = _RE_MATH_INLINE.sub(MATH_PLACEHOLDER, text)
    text = _RE_CMD_KEEP.sub(r"\1", text)
    text = _RE_CMD_DROP.sub("", text)
    text = _RE_CMD_SIMPLE.sub(" ", text)
    text = _RE_BRACES.sub("", text)
    return text


def strip_invisible_chars(text: str) -> str:
    """
    Supprime ou normalise les caractères Unicode invisibles.
    """
    text = text.replace("\u00a0", " ")
    text = _RE_INVISIBLE_SUPPRESS.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalise les espaces : collapse les espaces multiples, strip final.
    """
    text = strip_invisible_chars(text)
    text = _RE_SPACES.sub(" ", text)
    text = _RE_NEWLINES.sub("\n", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Normalise l'encodage Unicode en forme NFC.
    """
    return unicodedata.normalize("NFC", text)


def clean_text(text: str | None) -> str | None:
    """
    Pipeline complet de nettoyage d'un champ texte arXiv.
    """
    if not isinstance(text, str):
        # Gère None, float('nan') et tout autre type non-string
        return None
    text = html.unescape(text)
    text = clean_latex(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text
