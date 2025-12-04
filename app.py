import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi.applications import FastAPI
from fastapi.exceptions import HTTPException
from kink import di
from rdflib import Graph, URIRef
from SPARQLWrapper.Wrapper import TURTLE, SPARQLWrapper

from embedding_decoder import EmbeddingDecoder
from models import (
    AlignGraphRequest,
    AlignGraphResponse,
    AlignRequest,
    AlignResponse,
    HealthResponse,
    Suggestion,
)
from RankingModel import RankingModel
from utils import fetch_labels_for_suggestions

load_dotenv()

DECODER: EmbeddingDecoder | None = None
EMBEDDINGS: dict[str, np.ndarray] | None = None
ICD_EMBEDDINGS: dict[str, np.ndarray] | None = None
SNOMED_EMBEDDINGS: dict[str, np.ndarray] | None = None

DECODER_PATH = Path(os.getenv("DECODER_PATH", "TransE_128_dim_500_epochs_decoder.pth"))
EMBEDDINGS_PATH = Path(
    os.getenv("EMBEDDINGS", "decoders/TransE_128_dim_500_epochs_mapped.pkl")
)
ICD_EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS", "ICD_transe_epoch_100.pkl"))
SNOMED_EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS", "SNOMED_transe_epoch_100.pkl"))

SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "http://localhost:3030/dataset/sparql")
SPARQL_USERNAME = os.getenv("SPARQL_USERNAME", "")
SPARQL_PASSWORD = os.getenv("SPARQL_PASSWORD", "")

di["SPARQL_ENDPOINT"] = SPARQL_ENDPOINT
di["SPARQL_USERNAME"] = SPARQL_USERNAME
di["SPARQL_PASSWORD"] = SPARQL_PASSWORD

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models()
    yield
    await cleanup()


async def init_models():
    global MODEL

    logger = logging.getLogger("Stan")
    logger.setLevel(logging.DEBUG)

    MODEL = RankingModel(
        context_time_range=timedelta(days=365),
    )
    logger.info("Loaded model")


async def cleanup():
    global MODEL
    del MODEL


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Welcome to the Entity Alignment API"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_exists = DECODER_PATH.exists()
    model_loaded = DECODER is not None and EMBEDDINGS is not None
    return HealthResponse(
        status="healthy",
        model_exists=model_exists,
        model_loaded=model_loaded,
    )


def perform_alignment(
    graph: Graph, focus_node: str, input_code: str
) -> list[Suggestion]:
    """Perform alignment using the decoder and embeddings."""

    prediction = MODEL.predict(graph, URIRef(focus_node), input_code)
    return [
        Suggestion(value=str(s["value"]), confidence=s["confidence"])
        for s in prediction["suggestions"]
    ]  # type: ignore


def fetch_graph_from_sparql(graph_name: str) -> Graph:
    """Fetch graph triples from SPARQL endpoint."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    if SPARQL_USERNAME and SPARQL_PASSWORD:
        sparql.setCredentials(SPARQL_USERNAME, SPARQL_PASSWORD)

    # SPARQL CONSTRUCT query to get all triples from the named graph
    query = f"""
    CONSTRUCT {{
        ?s ?p ?o
    }}
    WHERE {{
        GRAPH <{graph_name}> {{
            ?s ?p ?o .
        }}
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(TURTLE)

    try:
        # CONSTRUCT returns RDF data directly, which we can parse into a Graph
        results = sparql.query().convert()
        graph = Graph()
        # The results should be bytes or string containing turtle data
        if isinstance(results, bytes):
            graph.parse(data=results.decode("utf-8"), format="turtle")
        else:
            graph.parse(data=str(results), format="turtle")

        return graph

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching graph from SPARQL: {str(e)}"
        )


@app.post("/align", response_model=AlignResponse)
def align(request: AlignRequest):
    """align missing properties for a focus node by fetching graph from SPARQL endpoint."""
    try:
        # Fetch graph from SPARQL endpoint
        graph = fetch_graph_from_sparql(request.graphName)

        suggestions = perform_alignment(graph, request.focus_node, request.input_code)

        fetch_labels_for_suggestions(suggestions, request.language or "en")

        for suggestion in suggestions:
            if not suggestion.value.startswith("http://snomed.info/id/"):
                suggestion.value = f"http://snomed.info/id/{suggestion.value}"

        return AlignResponse(
            id=request.id,
            graphName=request.graphName,
            focus_node=request.focus_node,
            input_code=request.input_code,
            suggestions=suggestions,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during imputation: {str(e)}"
        )


@app.post("/align_graph", response_model=AlignGraphResponse)
def align_graph(request: AlignGraphRequest):
    """align missing properties for a focus node using provided graph data."""
    try:
        # Parse the provided graph
        graph = Graph()
        graph.parse(data=request.graph, format=request.graph_format)

        # Perform imputation
        suggestions: list[Suggestion] = perform_alignment(
            graph, request.focus_node, request.input_code
        )

        for suggestion in suggestions:
            if not suggestion.value.startswith("http://snomed.info/id/"):
                suggestion.value = f"http://snomed.info/id/{suggestion.value}"

        return AlignGraphResponse(
            input_code=request.input_code,
            suggestions=suggestions,
            focus_node=request.focus_node,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during imputation: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
