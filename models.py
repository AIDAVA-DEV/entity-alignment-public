from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_exists: bool
    model_loaded: bool


class DisplayLabel(BaseModel):
    language: str
    value: str


# New models for imputation endpoints
class Suggestion(BaseModel):
    value: str
    confidence: float
    display: DisplayLabel | None = None


class AlignRequest(BaseModel):
    id: str
    graphName: str
    focus_node: str
    input_code: str
    language: str | None = None


class AlignResponse(BaseModel):
    id: str
    graphName: str
    focus_node: str
    input_code: str
    suggestions: list[Suggestion]


class AlignGraphRequest(BaseModel):
    graph: str
    graph_format: str = "ox-nt"
    focus_node: str
    input_code: str


class AlignGraphResponse(BaseModel):
    input_code: str
    focus_node: str
    suggestions: list[Suggestion]
