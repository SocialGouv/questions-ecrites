# Production Dockerfile généré par Atlas pour le POC QE.
# Si Victor commit son propre Dockerfile, celui-ci sera overwrite.
FROM python:3.12-slim AS builder
WORKDIR /app

# System deps : libpq-dev + gcc pour compiler psycopg2 depuis source.
# Alternative : utiliser psycopg2-binary dans pyproject.toml mais ça
# nécessite que Victor change sa dep.
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
      libpq-dev \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Poetry 2.x : supporte le format PEP 621 ([project] dans pyproject.toml).
# Le projet QE utilise ce format (pas l'ancien [tool.poetry]).
# pip >= 26.2.2 : son certifi embarqué ne fait plus confiance aux CA emSign
# compromises. Épinglé par hash car le pip de l'image de base les accepte encore.
RUN pip install --no-cache-dir "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
 && pip install --no-cache-dir poetry==2.1.3

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --no-root --only main

FROM python:3.12-slim
WORKDIR /app

# Runtime: juste libpq pour psycopg2, pas besoin des headers/compiler.
# ca-certificates : force la version à jour même si l'image de base est ancienne.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpq5 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy les libs Python installées
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy le code source
COPY . .

# Non-root user
RUN groupadd -g 1000 qe && useradd -m -u 1000 -g 1000 qe && chown -R qe:qe /app
USER 1000:1000

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
