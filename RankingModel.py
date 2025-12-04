import logging
from datetime import timedelta

from rdflib import Graph, URIRef  # type: ignore
from mappers.cpt_snomed_mapper import CptSnomedMapper
from mappers.ndc_snomed_mapper import NdcSnomedMapper
from mappers.loinc_snomed_mapper import LoincSnomedMapper
from mappers.icd_snomed_mapper import Icd9SnomedMapper
from mappers.mapper import MappingResult


class RankingModel:
    def __init__(self, context_time_range=timedelta(days=365), show_top_n=5):
        """
        Initialize the RankingModel with the given parameters.

        Parameters:
        - dict_1_to_1 (dict): A dictionary mapping ICD codes to a single SNOMED code.
        - dict_1_to_M (dict): A dictionary mapping ICD codes to multiple SNOMED codes.
        - nodes (dict): A dictionary of CodeTreeNode objects representing the SNOMED hierarchy.
        - SNOMED_embeddings (dict): A dictionary mapping SNOMED code URIs to their embeddings.
        - time_range (timedelta): The time range for context extraction with no direct connection to the focus_node (default is 365 days).
        """
        self.logger = logging.getLogger("RankingModel")
        self.SHOW_TOP_N = show_top_n

        self.__LOINC_SNOMED_mapper = LoincSnomedMapper()
        self.__CPT_SNOMED_mapper = CptSnomedMapper()
        self.__NDC_SNOMED_mapper = NdcSnomedMapper()

        # What is the time span of procedures with no pointer to an admission that will be considered
        # as part of an admission with a date within this range
        self.__ICD9_SNOMED_mapper = Icd9SnomedMapper(
            self.logger,
            self.__LOINC_SNOMED_mapper,
            self.__CPT_SNOMED_mapper,
            self.__NDC_SNOMED_mapper,
            context_time_range=context_time_range,
        )

    def predict(
        self, graph: Graph, focus_node: URIRef, code_to_predict: str, debug=False
    ):
        r"""
        returns a JSON-like dictionary with the structure:
        \{
            "suggestions":
            [
                {"value": \<predicted_code\>, "confidence": \<confidence_score\>},
                ...
            ],
            "comment": \<optional_comment\>
        }
        """
        prediction, comment = self.__make_prediction(
            graph, focus_node, code_to_predict, debug
        )
        return self.__create_JSON_response(
            prediction, focus_node, code_to_predict, comment
        )

    def __make_prediction(
        self, graph: Graph, focus_node: URIRef, code_to_predict: str, debug=False
    ) -> tuple[list[MappingResult], str]:
        vocabulary = self.__get_vocabulary_from_code(code_to_predict)
        code = code_to_predict.rsplit("/", 1)[-1].replace(".", "")
        if vocabulary == "UNKNOWN":
            self.logger.info(f"Code vocabulary of {code_to_predict} is UNKNOWN")
            return self.__predict_blank(
                f"The code vocabulary of {code_to_predict} is not recognized (not ICD-9, LOINC, NDC, or CPT)"
            )
        if vocabulary == "ICD-9":
            self.logger.info(
                f"Focus node {focus_node} is a ProblemCondition predicting ICD-9 code"
            )
            return self.__predict_icd_9_code(graph, focus_node, code)

        if vocabulary == "LOINC":
            self.logger.info(
                f"Focus node {focus_node} is a DrugPrescription predicting LOINC code"
            )
            return self.__predict_loinc_codes(code)

        if vocabulary == "NDC":
            self.logger.info(
                f"Focus node {focus_node} is a Measurement predicting NDC code"
            )
            return self.__predict_ndc_code(focus_node)

        if vocabulary == "CPT":
            self.logger.info(
                f"Focus node {focus_node} is a Procedure predicting CPT code"
            )
            return self.__predict_cpt_code(code)

        return self.__predict_blank(
            "The focus node is not of a supported type (ProblemCondition, Measurement, DrugPrescription, Procedure)"
        )

    def __create_JSON_response(
        self,
        ranking: list[MappingResult],
        focus_node: URIRef,
        code_to_predict: str,
        comment: str,
    ):
        response = {
            "focus_node": focus_node,
            "code": code_to_predict,
            "suggestions": [
                {
                    "value": result["snomed_code"],
                    "confidence": result["score"],
                    "is_AI_generated": result["is_ai"],
                }
                for result in ranking
            ],
            "comment": comment if hasattr(ranking, "comment") else "",
        }
        return response

    def __predict_blank(
        self,
        comment="",
    ) -> tuple[list[MappingResult], str]:
        return [], comment

    def __predict_icd_9_code(
        self,
        graph: Graph,
        focus_node: URIRef,
        code_to_predict: str,
    ) -> tuple[list[MappingResult], str]:
        try:
            ranking: list[MappingResult] = self.__ICD9_SNOMED_mapper.map(
                code_to_predict, graph, focus_node
            )

            if len(ranking) == 1:
                return ranking, r"This is a 1-to-1 mapping, so 100% confidence"  # type: ignore
            elif len(ranking) > self.SHOW_TOP_N:
                return (
                    ranking,
                    f"Showing only top {self.SHOW_TOP_N} predictions out of the {len(ranking)} that were mapped to",
                )
            else:
                return ranking, ""
        except Exception as e:
            self.logger.warning(str(e))
            return self.__predict_blank(str(e))

    def __predict_cpt_code(self, cpt_code: str) -> tuple[list[MappingResult], str]:
        try:
            ranking: list[MappingResult] = self.__CPT_SNOMED_mapper.map(
                cpt_code, Graph(), URIRef("")
            )
            if len(ranking) == 1:
                return ranking, r"This is a 1-to-1 mapping, so 100% confidence"  # type: ignore
            else:
                return ranking, ""
        except Exception as e:
            self.logger.warning(str(e))
            return self.__predict_blank(str(e))

    def __predict_ndc_code(self, ndc_code: str) -> tuple[list[MappingResult], str]:
        try:
            ranking: list[MappingResult] = self.__NDC_SNOMED_mapper.map(
                ndc_code, Graph(), URIRef("")
            )
            if len(ranking) == 1:
                return ranking, r"This is a 1-to-1 mapping, so 100% confidence"  # type: ignore
            else:
                return ranking, ""
        except Exception as e:
            self.logger.warning(str(e))
            return self.__predict_blank(str(e))

    def __predict_loinc_codes(self, loinc_code: str) -> tuple[list[MappingResult], str]:
        try:
            ranking: list[MappingResult] = self.__LOINC_SNOMED_mapper.map(
                loinc_code, Graph(), URIRef("")
            )
            # print(display_descriptions_of(snomed_codes, as_list=True))
            return (
                ranking,
                r"LOINC codes map to multiple SNOMED codes at the same time, each with 100% confidence. This LOINC code is equivalent to all returned SNOMED codes taken together, not individually. It could measure the present of an antigen in blood, which is represented by multiple SNOMED codes.",
            )  # type: ignore

        except Exception as e:
            self.logger.warning(str(e))
            return self.__predict_blank(str(e))

    def __get_vocabulary_from_code(self, code: str):
        if "icd-9" in code.lower():
            return "ICD-9"
        if "icd-10" in code.lower():
            return "ICD-10"
        if "ndc" in code.lower():
            return "NDC"
        if "loinc" in code.lower():
            return "LOINC"
        if "cpt" in code.lower():
            return "CPT"
        return "UNKNOWN"
