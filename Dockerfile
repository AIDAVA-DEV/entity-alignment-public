FROM ghcr.io/astral-sh/uv:bookworm-slim

ARG MODEL_PATH=all_data/generated/MIMIC_III/experiments/TransE_Linear/decoder_models/
ARG EMBEDDINGS_PATH=all_data/generated/MIMIC_III/experiments/TransE_Linear/embeddings/

ENV MODEL_PATH=${MODEL_PATH} \
    EMBEDDINGS_PATH=${EMBEDDINGS_PATH}

WORKDIR /app
COPY ./pyproject.toml ./uv.lock ./.python-version ./
RUN uv python install $(cat .python-version) && \
    uv sync --no-cache --compile-bytecode -q

COPY ./all_data ./all_data
COPY ./mappers /app/mappers
COPY  *.py ./


ENTRYPOINT ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000