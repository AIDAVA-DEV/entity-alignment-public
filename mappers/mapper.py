from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

from rdflib import Graph, URIRef # type: ignore

MappingResult = TypedDict(
    "MappingResult",
    {
        "snomed_code": str,
        "is_ai": bool,
        "score": float,
    },
)


class Mapper(ABC):
    base_data_path = Path("all_data/confidences/")

    @abstractmethod
    def map(self, code: str, graph: Graph, focus_node: URIRef) -> list[MappingResult]:
        """Maps the input code to a list of (snomed code, score) tuples.

        Args:
            graph (Graph): The RDF graph containing relevant data.
            focus_node (URIRef): The URI of the focus node in the graph.
            code (str): The input code to be mapped.

        Returns:
            list[MappingResult]: A list of mapping results containing SNOMED codes and their scores.
        """
        pass
