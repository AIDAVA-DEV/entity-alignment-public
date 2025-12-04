import logging

from rdflib.graph import Graph # type: ignore
from rdflib.term import URIRef # type: ignore

from mappers.mapper import Mapper, MappingResult
from utils import load_file_from_pkl


class CptSnomedMapper(Mapper):
    def __init__(self) -> None:
        self.logger = logging.getLogger("CPT_SNOMED_Mapper")
        self.__cpt_snomed_mapping_file_human = (
            self.base_data_path / "CPT_SNOMED_mapped_confidence.pkl"
        )
        self.__cpt_snomed_mapping_file_ai = (
            self.base_data_path / "CPT_SNOMED_unmapped_confidence.pkl"
        )
        file_exists = (
            self.__cpt_snomed_mapping_file_human.exists()
            and self.__cpt_snomed_mapping_file_ai.exists()
        )
        if not file_exists:
            self.logger.warning(
                f"CPT to SNOMED mapping files {self.__cpt_snomed_mapping_file_human} and {self.__cpt_snomed_mapping_file_ai} do not exist. CPT mappings will be empty."
            )
        self.CPT_to_SNOMED_dict_human = (
            load_file_from_pkl(self.__cpt_snomed_mapping_file_human) if file_exists else {}
        )
        self.CPT_to_SNOMED_dict_ai = (
            load_file_from_pkl(self.__cpt_snomed_mapping_file_ai) if file_exists else {}
        )

        super().__init__()

    def map(self, code: str, graph: Graph, focus_node: URIRef) -> list[MappingResult]:
        human_mappings = self.CPT_to_SNOMED_dict_human.get(code, [])
        ai_mappings = self.CPT_to_SNOMED_dict_ai.get(code, [])
        results: list[MappingResult] = []

        for (snomed_code, score) in human_mappings:
            results.append(
                {
                    "snomed_code": snomed_code,
                    "is_ai": False,
                    "score": score,
                }
            )
        for (snomed_code, score) in ai_mappings:
            results.append(
                {
                    "snomed_code": snomed_code,
                    "is_ai": True,
                    "score": score,
                }
            )
        return results
