"""
SNOMED Code Ranking Module
==========================

This module implements a non-parametric ranking of candidate SNOMED codes against a set of context codes
using precomputed embeddings. Distances are computed pairwise (Euclidean), percentile ranks are derived
per context, and scores are aggregated via mean. Confidence is assessed via coefficient of variation.

Assumptions:
- Embeddings are precomputed and stored as NumPy arrays or similar, keyed by SNOMED code (string).
- Lower distances indicate better semantic fit.
- Placeholders (e.g., file paths, code lists) are marked with comments for user customization.

Dependencies:
- numpy for vector operations and statistics.
- scipy for distance computation (optional; can use numpy alone).

Usage:
    python snomed_ranking.py
    or
    from snomed_ranking import rank_candidates
    results = rank_candidates(context_codes, candidate_codes, embeddings_path)
"""

from datetime import timedelta
import pandas as pd
from mappers.loinc_snomed_mapper import LoincSnomedMapper
from mappers.cpt_snomed_mapper import CptSnomedMapper
from mappers.ndc_snomed_mapper import NdcSnomedMapper
import numpy as np
from typing import List
import logging
from mappers.mapper import Mapper, MappingResult
from rdflib.graph import Graph  # type:ignore
from rdflib.namespace import Namespace, URIRef  # type:ignore
import os
from ContextExtractor import ContextExtractor
from utils import load_file_from_pkl, load_and_concat_embeddings, rdflib_to_nx


class Icd9SnomedMapper(Mapper):
    def __init__(
        self,
        logger: logging.Logger,
        LOINC_SNOMED_mapper: LoincSnomedMapper,
        CPT_SNOMED_mapper: CptSnomedMapper,
        NDC_SNOMED_mapper: NdcSnomedMapper,
        context_time_range=timedelta(days=365),
    ) -> None:
        self.context_time_range = context_time_range
        self.logger = logger
        self.snomed = Namespace("http://snomed.info/id/")
        self.graph = Graph()

        self.__LOINC_SNOMED_mapper = LOINC_SNOMED_mapper
        self.__CPT_SNOMED_mapper = CPT_SNOMED_mapper
        self.__NDC_SNOMED_mapper = NDC_SNOMED_mapper

        # We load all dictionaries and objects used by the mapper
        self.embedding_pattern = (
            "all_data/embeddings/Node2Vec_SNOMED/embeddings_part*.pkl"
        )
        self.SNOMED_embeddings = load_and_concat_embeddings(str(self.embedding_pattern))
        self.logger.debug("Loaded SNOMED embeddings")

        self.dict_load_path = "all_data/dicts"
        self.dict_1_to_1 = load_file_from_pkl(
            os.path.join(self.dict_load_path, "ICD_SNOMED_1_to_1.pkl")
        )
        self.dict_1_to_M = load_file_from_pkl(
            os.path.join(self.dict_load_path, "ICD_SNOMED_1_to_M.pkl")
        )
        self.logger.debug("Loaded ICD-9 to SNOMED mappings")

    def map(
        self,
        code: str,
        graph: Graph,
        focus_node: URIRef,
    ) -> list[MappingResult]:
        # If this is a 1-to-1 mapping, there is no need to even extract any context
        if code in self.dict_1_to_1:
            return self.__perform_1_to_1_prediction(code)
        elif code in self.dict_1_to_M:
            return self.__perform_1_to_M_prediction(graph, code, focus_node)
        else:
            raise ValueError(
                f"ICD code {code} not found in either 1-to-1 or 1-to-M mapping"
            )

    def __perform_1_to_1_prediction(self, icd9_code: str) -> list[MappingResult]:
        return [
            {"snomed_code": self.dict_1_to_1[icd9_code][0], "is_ai": False, "score": 1}
        ]

    def __perform_1_to_M_prediction(
        self, graph: Graph, icd9_code: str, focus_node: URIRef, debug=False
    ) -> List[MappingResult]:
        self.__try_load_graph(graph, debug)
        # returns: DataFrame with columns ["Type", "Node", "Code"]
        extracted_context: pd.DataFrame = (
            self.context_extractor.extract_context_from_graph(
                focus_node, time_range=self.context_time_range
            )
        )
        candidate_SNOMED_codes: list[str] = self.dict_1_to_M[icd9_code]
        context_SNOMED_codes: list[str] = self.__align_context_to_snomed(
            extracted_context
        )
        self.logger.debug(f"Aligned {len(context_SNOMED_codes)} context codes")

        return self.rank_candidates(context_SNOMED_codes, candidate_SNOMED_codes)

    def rank_candidates(
        self,
        context_codes: List[str],
        candidate_codes: List[str],
        aggregation: str = "mean",
    ) -> List[MappingResult]:
        """
        Main ranking function: Orchestrates loading, computation, and ranking.

        Args:
            context_codes (List[str]): List of context SNOMED codes.
            candidate_codes (List[str]): List of candidate SNOMED codes.
            aggregation (str): Score aggregation method ('mean' or 'median').

        Returns:
            Tuple[List[Tuple[str, float]], float, np.ndarray, np.ndarray]:
                - Ranked candidates: [(code, score), ...] sorted descending by score.
                - Confidence: Absolute confidence score.
                - Distances: Raw (m, n) distance matrix.
                - Scores: (m, n) percentile score matrix.
        """
        # Compute distances and scores
        distances = self.compute_pairwise_distances(context_codes, candidate_codes)
        scores = self.compute_percentile_scores(distances)
        agg_scores = self.aggregate_scores(scores, aggregation)
        # confidence = self.compute_confidence(scores)

        # Rank candidates
        ranked_indices = np.argsort(agg_scores)[::-1]  # Descending
        ranked_candidates: list[MappingResult] = []
        for idx in ranked_indices:
            ranked_candidates.append(
                {
                    "snomed_code": candidate_codes[idx],
                    "score": agg_scores[idx],
                    "is_ai": True,
                }
            )

        self.logger.debug(
            f"Ranking complete: Top candidate '{ranked_candidates[0]['snomed_code']}' with score {ranked_candidates[0]['score']:.4f}"
        )
        return ranked_candidates

    def __try_load_graph(self, graph: Graph, context_extractor_debug: bool = False):
        if self.graph and self.__are_graphs_equivalent(graph, self.graph):
            # If a prediction is done on the same graph, no need to reload it
            pass
        else:
            self.graph = graph
            self.context_extractor = ContextExtractor(
                rdflib_to_nx(graph, debug=context_extractor_debug), self.logger
            )

    def __are_graphs_equivalent(self, graph1, graph2):
        """
        Check if two RDF graphs are equivalent by verifying if they are isomorphic.
        This accounts for differences in triple ordering and blank node labels.
        """
        return graph1.isomorphic(graph2)

    def __align_context_to_snomed(self, extracted_context: pd.DataFrame) -> list[str]:
        # Map the aligned SNOMED codes to the extracted context
        aligned_LOINC_context = self.__align_LOINC_codes(extracted_context)
        aligned_NDC_context = self.__align_NDC_codes(extracted_context)
        aligned_CPT_context = self.__align_CPT_codes(extracted_context)

        aligned_context = pd.concat(
            [aligned_LOINC_context, aligned_NDC_context, aligned_CPT_context]
        )
        return self.__extract_context_codes(aligned_context)

    def __align_LOINC_codes(self, context_codes: pd.DataFrame) -> pd.DataFrame:
        measurement_type = URIRef(
            "https://biomedit.ch/rdf/sphn-ontology/sphn#Measurement"
        )
        filtered_LOINC_context = context_codes[
            context_codes["Type"] == measurement_type
        ]
        mapped_context = []
        for _, row in filtered_LOINC_context.iterrows():
            loinc_code = str(object=row["Code"]).split("/")[-1]
            try:
                mapping_results: list[MappingResult] = self.__LOINC_SNOMED_mapper.map(
                    loinc_code, Graph(), URIRef("")
                )
                # TODO: At this point, we can also include the confidence of the codes in the context dataframe if needed
                # print(display_descriptions_of(snomed_codes, as_list=True))
                for result in mapping_results:
                    # Ex. row: [URIRef("https://biomedit.ch/rdf/sphn-ontology/sphn#Measurement"), URIRef(), "124252"]
                    mapped_context.append(
                        (measurement_type, row["Node"], result["snomed_code"])
                    )
            except ValueError:
                # It is normal that some LOINC codes do not have a mapping to SNOMED
                continue

        return pd.DataFrame(mapped_context, columns=["Type", "Node", "Code"])

    def __align_NDC_codes(self, context_codes: pd.DataFrame) -> pd.DataFrame:
        prescription_type = URIRef(
            "https://biomedit.ch/rdf/sphn-ontology/sphn#DrugPrescription"
        )
        filtered_NDC_context = context_codes[context_codes["Type"] == prescription_type]
        mapped_context = []
        for _, row in filtered_NDC_context.iterrows():
            ndc_code = str(row["Code"]).split("/")[-1]
            try:
                mapping_results: List[MappingResult] = self.__NDC_SNOMED_mapper.map(
                    ndc_code, Graph(), URIRef("")
                )
                for result in mapping_results:
                    # print(display_description_of(snomed_code))
                    mapped_context.append(
                        (prescription_type, row["Node"], result["snomed_code"])
                    )
            except ValueError:
                # It is normal that some NDC codes do not have a mapping to SNOMED
                continue
        return pd.DataFrame(mapped_context, columns=["Type", "Node", "Code"])

    def __align_CPT_codes(self, extracted_context: pd.DataFrame) -> pd.DataFrame:
        procedure_type = URIRef(
            "https://biomedit.ch/rdf/sphn-ontology/sphn#Procedure"  # TODO: verify the link
        )
        filtered_CPT_context = extracted_context[
            extracted_context["Type"] == procedure_type
        ]
        mapped_context = []
        for _, row in filtered_CPT_context.iterrows():
            cpt_code = str(row["Code"]).split("/")[-1]
            try:
                mapping_results: List[MappingResult] = self.__CPT_SNOMED_mapper.map(
                    cpt_code, Graph(), URIRef("")
                )
                for result in mapping_results:
                    # print(display_description_of(snomed_code))
                    mapped_context.append(
                        (procedure_type, row["Node"], result["snomed_code"])
                    )
            except ValueError:
                # It is normal that some CPT codes do not have a mapping to SNOMED
                continue
        return pd.DataFrame(mapped_context, columns=["Type", "Node", "Code"])

    def __extract_context_codes(self, extracted_context: pd.DataFrame) -> List[str]:
        """
        Extracts a list of unique context codes from the provided DataFrame that have corresponding embeddings in SNOMED.

        Args:
            extracted_context (pd.DataFrame): The extracted context as a DataFrame with columns including "Code"
            (and optionally "Type" and "Node"). Only the "Code" column is utilized to filter codes present in SNOMED embeddings.

        Returns:
            List[str]: A list of unique codes (as strings) from the "Code" column that exist in the SNOMED embeddings.
        """
        context_codes = set()
        for code in extracted_context.Code:
            uri_ref = self.snomed[str(code)]
            if uri_ref in self.SNOMED_embeddings:
                context_codes.add(str(code))

        return list(context_codes)

    def compute_pairwise_distances(
        self,
        context_codes: List[str],
        candidate_codes: List[str],
    ) -> np.ndarray:
        """
        Compute Euclidean distances between all context-candidate pairs.

        Args:
            context_codes (List[str]): List of context SNOMED codes.
            candidate_codes (List[str]): List of candidate SNOMED codes.
            embeddings (Dict[str, np.ndarray]): Loaded embeddings.

        Returns:
            np.ndarray: (m, n) matrix of distances, where m = len(context_codes), n = len(candidate_codes).
        """
        m, n = len(context_codes), len(candidate_codes)
        distances = np.zeros((m, n))

        # Extract embedding vectors
        context_embs = np.stack(
            [self.SNOMED_embeddings[self.snomed[str(code)]] for code in context_codes]
        )
        candidate_embs = np.stack(
            [self.SNOMED_embeddings[self.snomed[str(code)]] for code in candidate_codes]
        )

        # Compute pairwise Euclidean distances
        for i in range(m):
            for j in range(n):
                distances[i, j] = np.linalg.norm(context_embs[i] - candidate_embs[j])

        self.logger.debug(f"Computed {m} x {n} distance matrix")
        return distances

    def compute_percentile_scores(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute percentile-based relative scores per context (row).

        For each context, rank candidates by distance (ascending), convert to percentile [0,1]
        where 1 = best (lowest distance).

        Args:
            distances (np.ndarray): (m, n) distance matrix.

        Returns:
            np.ndarray: (m, n) score matrix, s_{i,j} in [0,1].
        """
        m, n = distances.shape
        scores = np.zeros_like(distances)

        for i in range(m):
            # Get sorted indices for ranking (ascending distances)
            sorted_indices = np.argsort(distances[i])
            ranks = np.empty(n, dtype=int)
            ranks[sorted_indices] = np.arange(1, n + 1)  # 1-based ranks

            # Handle ties by averaging ranks (simple method; use scipy.stats.rankdata for advanced)
            unique_ranks, inverse = np.unique(ranks, return_inverse=True)
            for ur, inv in zip(unique_ranks, inverse):
                ranks[inv == ur] = ur  # For ties, this assigns same rank

            # Normalize to percentile: (rank - 1) / (n - 1), then invert for "better is higher"
            normalized_ranks = (ranks - 1) / (n - 1)
            scores[i] = 1 - normalized_ranks

        self.logger.debug(f"Computed percentile scores for {m} contexts")
        return scores

    def aggregate_scores(
        self, scores: np.ndarray, aggregation: str = "mean"
    ) -> np.ndarray:
        """
        Aggregate per-context scores to global candidate scores.

        Args:
            scores (np.ndarray): (m, n) score matrix.
            aggregation (str): 'mean' or 'median' for aggregation.

        Returns:
            np.ndarray: (n,) array of aggregated scores for candidates.
        """
        if aggregation == "mean":
            agg_scores = np.mean(scores, axis=0)
        elif aggregation == "median":
            agg_scores = np.median(scores, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        self.logger.debug(f"Aggregated scores using {aggregation}")
        return agg_scores

    def compute_confidence(
        self,
        scores: np.ndarray,
    ) -> float:
        """
        Compute absolute confidence for the ranking via coefficient of variation (CV) of per-context scores.

        Confidence near 1 indicates low variability (high robustness).

        Args:
            scores (np.ndarray): (m, n) score matrix.
            target (str): 'top' for top candidate's confidence, or 'all' for average CV across candidates.

        Returns:
            float: Confidence score in [0, 1].
        """
        m, n = scores.shape
        epsilon = 1e-6

        # Rank candidates by aggregated score
        agg_scores = np.mean(scores, axis=0)
        top_idx = np.argmax(agg_scores)
        top_row_scores = scores[:, top_idx]
        mu = np.mean(top_row_scores)
        sigma = np.std(top_row_scores)

        confidence = 1 - (sigma / (mu + epsilon))
        return confidence
