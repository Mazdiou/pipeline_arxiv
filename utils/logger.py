"""
Factory du logger applicatif : partagé par tous les modules du pipeline.

Principe de nommage des fichiers de log :
  - Si run_id fourni  → {log_dir}/{run_id}.log       (liaison log, rapport et historique)
  - Sinon             → {log_dir}/{timestamp}_{name}.log  (comportement par défaut)

Fonctionnalités :
  - Logger hiérarchique Python standard
  - StreamHandler → affichage console (niveau configurable)
  - FileHandler   → fichier de log (niveau configurable)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Formatter standard pour tous les handlers : horodatage + niveau + nom du logger + message
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logger(
    name: str = "arxiv.pipeline",
    log_dir: str = "logs",
    run_id: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Args:
        name          : Nom hiérarchique du logger (ex: 'arxiv.collector').
        log_dir       : Dossier de stockage des fichiers de log.
        run_id        : Identifiant unique du run (ex: '20260222_131406_collect_cs_AI').
        console_level : Niveau minimum pour la console (ex: logging.INFO).
        file_level    : Niveau minimum pour le fichier (ex: logging.DEBUG).

    Returns:
        logging.Logger : Logger configuré et prêt à l'emploi.
    """
    logger = logging.getLogger(name)

    # Évite l'ajout de handlers en double si le logger existe déjà
    if logger.handlers:
        return logger

    # Niveau global du logger : le filtrage fin est délégué aux handlers
    logger.setLevel(logging.DEBUG)

    # Handler console (affichage terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    # Handler fichier
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Nom du fichier selon la présence d'un run_id
    if run_id:
        log_file = log_dir_path / f"{run_id}.log"
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"{timestamp}_{name.replace('.', '_')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)

    logger.info(f"Logger '{name}' initialisé → {log_file}")
    return logger
