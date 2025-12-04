import glob
import logging
import math
import os
import pickle

import networkx as nx
import pandas as pd
from kink import di
from rdflib import Graph  # type:ignore
from SPARQLWrapper.Wrapper import JSON, SPARQLWrapper
from tqdm import tqdm

from models import DisplayLabel, Suggestion


def load_and_concat_embeddings(path_pattern):
    """Load smaller dictionaries and concatenate them into one."""
    concatenated_embeddings = {}
    files = glob.glob(path_pattern)
    for file in files:
        with open(file, "rb") as f:
            sub_dict = pickle.load(f)
            concatenated_embeddings.update(sub_dict)
    return concatenated_embeddings


def load_embeddings_from_files(path_pattern):
    embeddings_dict = {}
    files = glob.glob(path_pattern)
    for file in files:
        temp_dict = load_file_from_pkl(file)
        embeddings_dict.update(temp_dict)
    return embeddings_dict


def count_element_lengths(array):
    return pd.Series([len(str(element)) for element in array]).value_counts()


def count_dic_value_lengths(dic):
    return pd.Series([len(value) for value in dic.values()]).value_counts()


def count_df_group_lengths(df, group_col, list_col):
    return count_dic_value_lengths(
        df.groupby(group_col)
        .apply(lambda x: x[list_col].tolist(), include_groups=False)
        .to_dict()
    )


def load_file_from_pkl(file_path):
    """
    Load a dictionary from a pickle file.
    """

    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_file_to_pkl(data, file_path):
    """
    Save a dictionary to a pickle file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_dict_from_files(path_pattern):
    embeddings_dict = {}
    files = glob.glob(path_pattern)
    for file in files:
        temp_dict = load_file_from_pkl(file)
        embeddings_dict.update(temp_dict)
    return embeddings_dict


def split_and_save_dict(embeddings, save_path, filename, n):
    # Split the dictionary into n equal parts
    items = list(embeddings.items())
    part_size = math.ceil(len(items) / n)
    parts = [dict(items[i : i + part_size]) for i in range(0, len(items), part_size)]

    # Save each part using a for loop
    for i, part in enumerate(parts):
        save_file_to_pkl(part, os.path.join(save_path, f"{filename}{i}.pkl"))


def rdflib_to_nx(rdflib_graph: Graph, debug=False):
    G = nx.DiGraph()
    if debug:
        for s, p, o in tqdm(
            rdflib_graph,
            desc="Converting rdflib graph to networkx graph",
            total=len(rdflib_graph),
        ):
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, relation=p)
    else:
        for s, p, o in rdflib_graph:
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, relation=p)
    return G


LABEL_QUERY_TEMPLATE = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?label WHERE {{
    <{entity_uri}> rdfs:label|skos:prefLabel ?label .
    FILTER (lang(?label) = "{language}" || lang(?label) = "en")
}}
ORDER BY DESC(lang(?label) = "{language}")
LIMIT 1
"""


def fetch_labels_for_suggestions(suggestions: list[Suggestion], language: str):
    logger = logging.getLogger("LabelFetcher")
    SPARQL_ENDPOINT: str = di["SPARQL_ENDPOINT"]
    SPARQL_USERNAME: str = di["SPARQL_USERNAME"]
    SPARQL_PASSWORD: str = di["SPARQL_PASSWORD"]
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    if SPARQL_USERNAME and SPARQL_PASSWORD:
        sparql.setCredentials(SPARQL_USERNAME, SPARQL_PASSWORD)

    for suggestion in suggestions:
        query = LABEL_QUERY_TEMPLATE.format(
            entity_uri=suggestion.value, language=language
        )
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query()._convertJSON()
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                label_value = bindings[0]["label"]["value"]
                suggestion.display = DisplayLabel(language=language, value=label_value)
        except Exception as e:
            logger.error(f"Error fetching label for {suggestion.value}: {str(e)}")
