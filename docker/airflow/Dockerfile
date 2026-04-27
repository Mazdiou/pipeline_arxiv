# docker/airflow/Dockerfile
#
# Image Airflow personnalisée pour l'ArXiv Pipeline.
#
# Base : apache/airflow:2.9.3-python3.11
# Ajouts :
#   - Dépendances système pour les extensions C (numpy, pyarrow, hdbscan…)
#   - PyTorch CPU (version légère, sans CUDA)
#   - Toutes les dépendances Python du projet (collector, processor, enricher, dw)
#   - python-dotenv (chargement .env)
#   - requests (dashboard → API REST Airflow)

FROM apache/airflow:2.9.3-python3.11

# ── Dépendances système ──────────────────────────────────────────────────────
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ───────────────────────────────────────────────────────
USER airflow

# Copie des fichiers requirements (couche Docker cachée si inchangés)
COPY --chown=airflow:root requirements/base.txt      /tmp/req/base.txt
COPY --chown=airflow:root requirements/processor.txt /tmp/req/processor.txt
COPY --chown=airflow:root requirements/nlp.txt       /tmp/req/nlp.txt
COPY --chown=airflow:root requirements/dw.txt        /tmp/req/dw.txt

# PyTorch CPU d'abord — évite de télécharger la version CUDA (~3 Go)
# via sentence-transformers / bertopic qui en dépendent
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Dépendances projet
RUN pip install --no-cache-dir \
    -r /tmp/req/base.txt \
    -r /tmp/req/processor.txt \
    -r /tmp/req/nlp.txt \
    -r /tmp/req/dw.txt \
    python-dotenv \
    requests

# Modèle scispaCy — téléchargé séparément (non disponible sur PyPI)
# ~900 MB — couche Docker dédiée pour optimiser le cache
RUN pip install --no-cache-dir \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
