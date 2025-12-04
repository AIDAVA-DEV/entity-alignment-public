import logging

from rdflib.graph import Graph # type: ignore
from rdflib.term import URIRef # type: ignore

from mappers.mapper import Mapper, MappingResult
from utils import load_file_from_pkl


class LoincSnomedMapper(Mapper):
    def __init__(self) -> None:
        self.logger = logging.getLogger("LOINC_SNOMED_Mapper")
        self.__loinc_snomed_mapping_file_human = (
            self.base_data_path / "LOINC_SNOMED_mapped_confidence.pkl"
        )
        file_exists = (
            self.__loinc_snomed_mapping_file_human.exists()
        )
        if not file_exists:
            self.logger.warning(
                f"LOINC to SNOMED mapping file {self.__loinc_snomed_mapping_file_human} does not exist. LOINC mappings will be empty."
            )
        self.LOINC_to_SNOMED_dict_human = (
            load_file_from_pkl(self.__loinc_snomed_mapping_file_human) if file_exists else {}
        )

        super().__init__()

    def map(self, code: str, graph: Graph, focus_node: URIRef) -> list[MappingResult]:
        human_mappings = self.LOINC_to_SNOMED_dict_human.get(code, [])
        results: list[MappingResult] = []

        for (snomed_code, score) in human_mappings:
            results.append(
                {
                    "snomed_code": snomed_code,
                    "is_ai": False,
                    "score": score,
                }
            )
        return results
