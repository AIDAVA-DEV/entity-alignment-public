from datetime import datetime, timedelta
from logging import Logger
import pandas as pd
from networkx import DiGraph
from rdflib import RDF, Namespace, URIRef  # type: ignore

sphn = Namespace("https://biomedit.ch/rdf/sphn-ontology/sphn#")
purl = Namespace("http://purl.bioontology.org/ontology/")
snomed = Namespace("http://snomed.info/id/")
AIDAVA = Namespace("https://biomedit.ch/rdf/sphn-ontology/AIDAVA/")


class NXGraphQuery:
    def __init__(self, nx_graph: DiGraph) -> None:
        self.nx_graph = nx_graph

    def is_connected_to(
        self, source: URIRef, relation: URIRef, destination: URIRef
    ) -> bool:
        connections = self.nx_graph.succ[source]
        if destination in list(connections):
            if relation == connections[destination]["relation"]:
                return True
        return False

    def get_successors(
        self, source: URIRef, relation: URIRef | None = None
    ) -> list[tuple[URIRef, URIRef, URIRef]]:
        """Return (source, relation, target)"""
        connected_nodes = []
        connections = self.nx_graph.succ.get(source, {})
        for target, attrs in connections.items():
            if relation is None or attrs["relation"] == relation:
                connected_nodes.append((source, attrs["relation"], target))
        return connected_nodes

    def get_predecessors(
        self, target: URIRef, relation: URIRef | None = None
    ) -> list[tuple[URIRef, URIRef, URIRef]]:
        """Return (source, relation, target)"""
        connected_nodes = []
        connections = self.nx_graph.pred.get(target, {})
        for source, attrs in connections.items():
            if relation is None or attrs["relation"] == relation:
                connected_nodes.append((source, attrs["relation"], target))
        return connected_nodes

    def has_successors(self, source: URIRef, relation: URIRef) -> bool:
        return len(self.get_successors(source, relation)) > 0

    def has_predecessors(self, target: URIRef, relation: URIRef) -> bool:
        return len(self.get_predecessors(target, relation)) > 0


class ContextExtractor:
    def __init__(self, nx_graph: DiGraph, logger: Logger) -> None:
        self.__nx_graph = nx_graph
        self.__query_nx_graph = NXGraphQuery(nx_graph)
        self.logger = logger

    def __get_codes(self, nodes: list[URIRef], code_property: URIRef) -> pd.DataFrame:
        """
        Return a dictionary: node RDF.types --> list of codes from each of those node types
        Filters only the codes that have the specified code_property
        """
        codes: list[tuple[URIRef, URIRef, URIRef]] = []
        for node in nodes:
            results: list[tuple[URIRef, URIRef, URIRef]] = (
                self.__query_nx_graph.get_successors(node, code_property)
            )
            if not len(results) == 0:
                rdf_type: URIRef = self.__query_nx_graph.get_successors(node, RDF.type)[
                    0
                ][2]
                codes.extend((rdf_type, node, result[2]) for result in results)
        return pd.DataFrame(codes, columns=["Type", "Node", "Code"])

    def __get_patient_nodes(self, patient: URIRef) -> list[URIRef]:
        context_nodes = self.__query_nx_graph.get_predecessors(
            patient, AIDAVA["hasPatient"]
        )
        context_nodes = [pred for (pred, _, _) in context_nodes]
        return context_nodes

    def __get_admission_nodes(self, admission: URIRef) -> list[URIRef]:
        context_nodes = self.__query_nx_graph.get_predecessors(
            admission, sphn["hasAdministrativeCase"]
        )
        context_nodes = [pred for (pred, _, _) in context_nodes]
        return context_nodes

    def __get_context_with_admission(self, admission: URIRef) -> pd.DataFrame:
        nodes_pointing_to_admission = self.__get_admission_nodes(admission)
        return self.__get_codes(nodes_pointing_to_admission, sphn["hasCode"])

    def __get_context_with_patient(self, patient: URIRef) -> pd.DataFrame:
        nodes_pointing_to_patient = self.__get_patient_nodes(patient)
        return self.__get_codes(nodes_pointing_to_patient, sphn["hasCode"])

    def __add_context_codes(
        self, all_context_codes: pd.DataFrame, new_codes: pd.DataFrame
    ) -> pd.DataFrame:
        if not all_context_codes.columns.equals(new_codes.columns):
            raise ValueError(
                f"DataFrames do not have matching columns. All context codes columns: {all_context_codes.columns}, new codes columns: {new_codes.columns}"
            )
        return pd.concat([all_context_codes, new_codes], ignore_index=True)

    def __get_filtered_context(
        self, must_miss_admission=True, must_miss_patient=True
    ) -> pd.DataFrame:
        """
        :missing_admission: - must be missing admission but can have patient
        :missing_patient: - must be missing patient but can have admission<br>
        If both are True, must be missing both admission and patient<br>
        If both are False, everything is selected"""
        # Has code (Procedure, ProblemCondition, Measurement, DrugPrescription, AdministrativeGender) but no admissions or patients
        procedures = self.__query_nx_graph.get_predecessors(sphn["Procedure"], RDF.type)
        problem_conditions = self.__query_nx_graph.get_predecessors(
            sphn["ProblemCondition"], RDF.type
        )
        measurements = self.__query_nx_graph.get_predecessors(
            sphn["Measurement"], RDF.type
        )
        drug_prescriptions = self.__query_nx_graph.get_predecessors(
            sphn["DrugPrescription"], RDF.type
        )
        administrative_genders = self.__query_nx_graph.get_predecessors(
            sphn["AdministrativeGender"], RDF.type
        )

        nodes_with_code = []
        for node in (
            procedures
            + problem_conditions
            + measurements
            + drug_prescriptions
            + administrative_genders
        ):
            # If the node has no admissions or patients, add it to the list
            passes_requirements = True
            if must_miss_admission:
                if must_miss_patient:
                    passes_requirements = not self.__query_nx_graph.has_successors(
                        node[0], sphn["hasAdministrativeCase"]
                    ) and not self.__query_nx_graph.has_successors(
                        node[0], AIDAVA["hasPatient"]
                    )
                else:
                    passes_requirements = not self.__query_nx_graph.has_successors(
                        node[0], sphn["hasAdministrativeCase"]
                    )
            elif must_miss_patient:
                passes_requirements = not self.__query_nx_graph.has_successors(
                    node[0], AIDAVA["hasPatient"]
                )
            if passes_requirements:
                codes: list[tuple[URIRef, URIRef, URIRef]] = (
                    self.__query_nx_graph.get_successors(node[0], sphn["hasCode"])
                )
                if len(codes) > 0:
                    code = codes[0][2]
                    dtype: URIRef = node[2]
                    nodes_with_code.append((dtype, node[0], code))
        return pd.DataFrame(nodes_with_code, columns=["Type", "Node", "Code"])

    def __filter_context_within_time(
        self, context: pd.DataFrame, time: datetime, delta: timedelta
    ) -> pd.DataFrame:
        """
        Filter any context dataframe to include only rows where the node has a date within the specified time range.
        """
        filtered_context = []
        for _, row in context.iterrows():
            dates = self.__query_nx_graph.get_successors(
                row["Node"], sphn["hasMeasurementDateTime"]
            )
            if len(dates) > 0:
                date = dates[0][2].toPython()
                if isinstance(date, datetime) and time - delta <= date <= time + delta:
                    filtered_context.append(row)
        return pd.DataFrame(filtered_context, columns=context.columns)

    def __get_filtered_context_within_time(
        self,
        time: datetime,
        delta: timedelta,
        must_miss_admission=True,
        must_miss_patient=True,
    ) -> pd.DataFrame:
        orphaned_context = self.__get_filtered_context(
            must_miss_admission, must_miss_patient
        )
        return self.__filter_context_within_time(orphaned_context, time, delta)

    def __get_patient_context_within_time(
        self, patient: URIRef, time: datetime, delta: timedelta
    ) -> pd.DataFrame:
        """
        Get context nodes connected to a patient within a specified time range.
        """
        context_codes = self.__get_context_with_patient(patient)
        filtered_context = self.__filter_context_within_time(context_codes, time, delta)
        return filtered_context

    def __get_node_date(self, node: URIRef) -> datetime | None:
        dates = self.__query_nx_graph.get_successors(
            node, sphn["hasMeasurementDateTime"]
        )
        if len(dates) > 0:
            # We only expect a single correctly formatted date
            date = dates[0][2].toPython()
            if type(date) is datetime:
                self.logger.debug(f"- Node {node} has date {date}")
                return date
        return None

    def __focus_is_condition(self, focus_node: URIRef) -> bool:
        return (
            self.__query_nx_graph.get_successors(focus_node, RDF.type)[0][2]
            == sphn["ProblemCondition"]
        )

    def __focus_is_procedure(self, focus_node: URIRef) -> bool:
        return (
            self.__query_nx_graph.get_successors(focus_node, RDF.type)[0][2]
            == sphn["Procedure"]
        )

    def __focus_is_measurement(self, focus_node: URIRef) -> bool:
        return (
            self.__query_nx_graph.get_successors(focus_node, RDF.type)[0][2]
            == sphn["Measurement"]
        )

    def __focus_is_drug_prescription(self, focus_node: URIRef) -> bool:
        return (
            self.__query_nx_graph.get_successors(focus_node, RDF.type)[0][2]
            == sphn["DrugPrescription"]
        )

    def __get_patient(self, focus_node: URIRef) -> URIRef | None:
        patient = self.__query_nx_graph.get_successors(focus_node, AIDAVA["hasPatient"])
        if len(patient) > 0:
            patient = patient[0][2]
            return patient
        return None

    def __get_admission(self, focus_node: URIRef) -> URIRef | None:
        admission = self.__query_nx_graph.get_successors(
            focus_node, sphn["hasAdministrativeCase"]
        )
        if len(admission) > 0:
            admission = admission[0][2]
            return admission
        return None

    def __get_unique_admissions(self) -> set[URIRef]:
        admissions = self.__query_nx_graph.get_predecessors(
            sphn["AdministrativeCase"], RDF.type
        )
        admissions = set([adm[0] for adm in admissions])
        return admissions

    def extract_context_from_graph(
        self, focus_node: URIRef, time_range: timedelta
    ) -> pd.DataFrame:
        """
        Extract context codes for a given focus node based on the following logic:
        1. If the focus node has an administrative case (admission):
        - Add all context directly connected to that admission.
        - If the admission has a date, add all orphaned context with no admission that has a date within +- 1 year of this admission's date.
        2. If the focus node does not have an administrative case but has a patient:
        - If the focus node has a date, add all context connected to that patient within +- 1 year of this focusNode's date.
        - If the focus node does not have a date, add all context connected to that patient.
        3. If the focus node has neither an administrative case nor a patient:
        - If the focus node has a date, add all orphaned context within +- 1 year of that date.
        - If the focus node does not have a date, add all orphaned context.

        returns: DataFrame with columns ["Type", "Node", "Code"]
        """
        all_context_codes = pd.DataFrame(columns=["Type", "Node", "Code"])

        # Check if focus node has AdministrativeCase
        admission = self.__get_admission(focus_node)
        if admission is not None:
            self.logger.debug("Focus node has administrative case")
            # Take all context directly connected to that admission
            codes = self.__get_context_with_admission(admission)
            self.logger.debug(
                f"- Found {len(codes)} nodes connected with hasCode through admission {admission}"
            )
            all_context_codes = self.__add_context_codes(all_context_codes, codes)

            # Check, if admission has a date
            admission_date = self.__get_node_date(admission)
            if admission_date is not None:
                self.logger.debug("Focus node has measurement date time")
                self.logger.debug(
                    f"Checking for context missing admission within +- 1 year of admission date {admission_date}"
                )
                # If yes, add all context with no admission that has a date within +- 1 year of this admission's date
                codes = self.__get_filtered_context_within_time(
                    admission_date, time_range, must_miss_patient=False
                )
                self.logger.debug(
                    f"- Found {len(codes)} context nodes missing admission within +- {time_range} of admission date {admission_date}"
                )
                all_context_codes = self.__add_context_codes(all_context_codes, codes)
            else:
                self.logger.debug("Admission does not have measurement date time")
        else:
            self.logger.debug("Focus node does not have administrative case")
            # Check if focus node has Patient
            patient = self.__get_patient(focus_node)
            if patient is not None:
                self.logger.debug("Focus node has patient")
                # Check if focusNode has a date
                focus_node_date = self.__get_node_date(focus_node)
                if focus_node_date is not None:
                    self.logger.debug("Focus node has measurement date time")
                    self.logger.debug(
                        f"Checking for context with patient within +- 1 year of focus node date {focus_node_date}"
                    )
                    # If yes, add all context connected to that patient within +- 1 year of this focusNode's date
                    codes = self.__get_patient_context_within_time(
                        patient, focus_node_date, time_range
                    )
                    all_context_codes = self.__add_context_codes(
                        all_context_codes, codes
                    )
                    self.logger.debug(
                        f"- Found {len(codes)} context nodes with patient within +- {time_range} of focus node date {focus_node_date}"
                    )
                else:
                    self.logger.debug("Focus node does not have measurement date time")
                    self.logger.debug("Getting all context connected to that patient")
                    # If no, add all context connected to that patient
                    codes = self.__get_context_with_patient(patient)
                    all_context_codes = self.__add_context_codes(
                        all_context_codes, codes
                    )
            else:
                self.logger.debug("Focus node does not have patient")
                # If the focus node hasDate, get orphaned context within +- 1 year of that date
                focus_node_date = self.__get_node_date(focus_node)
                if focus_node_date is not None:
                    self.logger.debug("Focus node has measurement date time")
                    self.logger.debug(
                        "Checking for context within +- 1 year of focus node date"
                    )
                    codes = self.__get_filtered_context_within_time(
                        focus_node_date,
                        time_range,
                        must_miss_admission=False,
                        must_miss_patient=False,
                    )
                    all_context_codes = self.__add_context_codes(
                        all_context_codes, codes
                    )
                else:
                    self.logger.debug("Focus node does not have measurement date time")
                    self.logger.debug("Getting all context")
                    # If no, add all orphaned context
                    codes = self.__get_filtered_context(
                        must_miss_admission=False, must_miss_patient=False
                    )
                    all_context_codes = self.__add_context_codes(
                        all_context_codes, codes
                    )

        self.logger.debug(f"Total context codes found: {len(all_context_codes)}")
        return all_context_codes
