from utils import load_file_from_pkl
from tqdm import tqdm

# from CodeTreeNode import CodeTreeNode, is_descendant, get_all_descendant_nodes
class Descriptions:
    def __init__(self) -> None:
        dict_dir = "all_data/dicts/"
        self.relationship_desc_dict = load_file_from_pkl(dict_dir + "SNOMED_relationships_desc.pkl")
        self.SNOMED_desc_dict = load_file_from_pkl(dict_dir + "SNOMED_descriptions.pkl")
        
    def __get_relationship_description(self, code):
        return self.relationship_desc_dict[code]

    def get_description_of(self, code):
        return self.SNOMED_desc_dict.get(code)

    def display_description_of(self, code):
        if code in self.relationship_desc_dict:
            desc = self.__get_relationship_description(code) or ""
        else:
            desc = self.get_description_of(code)
        return str(code) + ": " + desc

    def display_descriptions_of(self, codes, as_list=True, ignore_missing=True):
        if as_list:
            return [self.display_description_of(code) for code in codes if code in self.SNOMED_desc_dict or not ignore_missing]
        return "\n".join([self.display_description_of(code) for code in codes if code in self.SNOMED_desc_dict or not ignore_missing])


class CodeTreeNode:
    def __init__(
        self,
        nodes: dict[int, "CodeTreeNode"],
        code: int,
        parent_codes: list[int],
        children_codes: list[int],
        outward_relationships: dict | None = None,
        inward_relationships: dict | None = None,
    ):
        self.descriptions = Descriptions()
        self.code = code
        self.parent_codes: list[int] = parent_codes
        self.children_codes: list[int] = children_codes
        self.outward_relationships: dict[int, list[int]] = (
            outward_relationships if outward_relationships is not None else {}
        )
        self.inward_relationships: dict[int, list[int]] = (
            inward_relationships if inward_relationships is not None else {}
        )
        self.nodes: dict[int, "CodeTreeNode"] = (
            nodes  # Reference to the global nodes dictionary
)

    def desc(self):
        return self.descriptions.get_description_of(self.code)

    def __repr__(self) -> str:
        desc = self.descriptions.display_description_of(self.code)
        parents = [self.descriptions.display_description_of(p) for p in self.parent_codes]
        children = [self.descriptions.display_description_of(c) for c in self.children_codes]
        outward_relationships = {
            self.descriptions.display_description_of(k): [self.descriptions.display_description_of(v) for v in values]
            for k, values in self.outward_relationships.items()
        }
        inward_relationships = {
            self.descriptions.display_description_of(k): [self.descriptions.display_description_of(v) for v in values]
            for k, values in self.inward_relationships.items()
        }
        return (
            f"Code: {self.code}\n"
            f"Description: {desc}\n"
            f"Parents: {parents}\n"
            f"Children: {children}\n"
            f"Outward relationships: {outward_relationships}\n"
            f"Inward relationships: {inward_relationships}"
        )

    def short_repr(self):
        desc = self.descriptions.display_description_of(self.code)
        print(f"{desc}\nParents: {self.parent_codes}\nChildren: {self.children_codes}")

    def children_codes_desc(self) -> dict[int, str]:
        return {code: self.descriptions.get_description_of(code) for code in self.children_codes}
    def parent_codes_desc(self) -> dict[int, str]:
        return {code: self.descriptions.get_description_of(code) for code in self.parent_codes}

    def go_to_child(self, child_code) -> "CodeTreeNode":
        if child_code not in self.children_codes:
            raise ValueError("Child code not found in this node's children")
        return self.nodes[child_code]

    def go_to_parent(self, parent_code) -> "CodeTreeNode":
        if parent_code not in self.parent_codes:
            raise ValueError("Parent code not found in this node's parents")
        return self.nodes[parent_code]

    def __is_descendant(self, child_node, parent_node) -> bool:
        """
        Check if a node is a descendant of another node in the hierarchy.

        Parameters:
        - child_node: The node to check if it is a descendant.
        - parent_node: The node to check if it is an ancestor.

        Returns:
        - True if child_node is a descendant of parent_node, False otherwise.
        """
        visited = set()
        queue = [child_node]

        while queue:
            current_node: CodeTreeNode = queue.pop(0)
            if current_node in visited:
                continue
            visited.add(current_node)

            # Check if the current node is the parent node
            if current_node == parent_node:
                return True

            # Add the parents of the current node to the queue
            queue.extend(
                [current_node.nodes[code] for code in current_node.parent_codes]
            )

        return False

    def is_child_of(self, parent_node):  # type: ignore
        return self.__is_descendant(self, parent_node)

    def is_parent_of(self, child_node):  # type: ignore
        return self.__is_descendant(child_node, self)

    @staticmethod
    def pickle_nodes(nodes: dict[int, "CodeTreeNode"], file_path: str):
        import pickle

        with open(file_path, "wb") as file:
            pickle.dump(nodes, file)

    @staticmethod
    def unpickle_nodes(file_path: str) -> dict[int, "CodeTreeNode"]:
        import pickle
        import sys

        sys.modules["__main__"].__setattr__("CodeTreeNode", CodeTreeNode)

        with open(file_path, "rb") as file:
            return pickle.load(file)


def is_descendant(child_node: CodeTreeNode, parent_node: CodeTreeNode) -> bool:
    """
    Check if a node is a descendant of another node in the hierarchy.

    Parameters:
    - child_node: The node to check if it is a descendant.
    - parent_node: The node to check if it is an ancestor.

    Returns:
    - True if child_node is a descendant of parent_node, False otherwise.
    """
    visited = set()
    queue = [child_node]

    while queue:
        current_node: CodeTreeNode = queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        # Check if the current node is the parent node
        if current_node == parent_node:
            return True

        # Add the parents of the current node to the queue
        queue.extend([current_node.nodes[code] for code in current_node.parent_codes])

    return False


# Select all descendant nodes of a given node
def get_all_descendant_nodes(parent_node: CodeTreeNode) -> list[int]:
    descendants = []
    for node in tqdm(
        parent_node.nodes.values(), desc=f"Finding descendants of {parent_node.code}"
    ):
        if is_descendant(node, parent_node):
            descendants.append(node.code)
    return descendants


# Recursive to be used on small datasets
def find_all_descendants(starting_node: CodeTreeNode) -> list[int]:
    descendants = set()
    visited = set()
    queue = [starting_node.code]

    while queue:
        current_code = queue.pop(0)
        if current_code in visited:
            continue
        visited.add(current_code)
        children = starting_node.nodes[current_code].children_codes
        descendants.update(children)
        queue.extend(children)

    return list(descendants)


def __check_if_parent_of_current_parent(
    code: CodeTreeNode, current_parent_codes: list[CodeTreeNode]
) -> list[CodeTreeNode]:
    parent_of = []
    for parent_code in current_parent_codes:
        if is_descendant(parent_code, code):
            parent_of.append(parent_code)
    return parent_of


def find_top_parents(
    candidate_nodes: list[CodeTreeNode], debug=False
) -> list[CodeTreeNode]:
    current_parent_codes = [candidate_nodes[0]]
    for code in candidate_nodes[1:]:
        if any(
            is_descendant(code, parent_code) for parent_code in current_parent_codes
        ):
            if debug:
                print(code, "is a descendant")
            continue
        else:
            parent_of = __check_if_parent_of_current_parent(code, current_parent_codes)
            if len(parent_of) == 0:
                if debug:
                    print(f"Code {code} is part of a different tree")
                current_parent_codes.append(code)
            else:
                if debug:
                    print(
                        f"Code: {code} Parent of {parent_of}, current parent codes {current_parent_codes}"
                    )
                current_parent_codes.append(code)
                for parent_code in parent_of:
                    if debug:
                        print(code, "is a parent of a current parent")
                    current_parent_codes.remove(parent_code)
    return current_parent_codes


# nodes = load_file_from_pkl("all_data/generated/MIMIC_III/dicts/SNOMED_tree_nodes.pkl")
