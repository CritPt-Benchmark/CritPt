import ast
import logging
import re
from typing import List

def extract_names(node: ast.AST) -> List[str]:
    """
    Recursively gather bare variable names bound by an assignment target.
    Returns a list because tuple/list unpacking can bind many names at once.
    Ignores attributes (`obj.x`), subscripts (`arr[i]`), etc.
    """
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        return [node.id]

    # Tuple, list, or starred unpacking:  (a, *rest) = ...
    if isinstance(node, (ast.Tuple, ast.List)):
        names = []
        for elt in node.elts:
            names.extend(extract_names(elt))
        return names

    # *rest = ...  →  node is ast.Starred; its .value holds the real target
    if isinstance(node, ast.Starred):
        return extract_names(node.value)

    # Anything else (Attribute, Subscript, etc.) does not bind a *new* name
    return []

def assigned_names(stmt: ast.AST) -> List[str]:
    """
    Given a statement (Assign, AugAssign, AnnAssign, NamedExpr) return
    all plain variable names it assigns to.
    """
    if isinstance(stmt, ast.Assign):
        names = []
        for tgt in stmt.targets:
            names.extend(extract_names(tgt))
        return names

    if isinstance(stmt, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
        return extract_names(stmt.target)

    return []          # not an assignment


def extract_python_script(response, allow_entirety_as_script=True, ignore_imports=True):
    # We will extract the python script from the response
    python_script = None
    if '```' in response:
        if '```python' in response:
            python_script = response.split("```python")[1].split("```")[0]
        elif re.search(r'```\s*\n', response):
            python_script = response.split('```')[1].split('```')[0]
        else:
            python_script = None

    if python_script is None:
        if allow_entirety_as_script:
            # logging.info("[extract_python_script] There is no codeblock. Assuming the entirety as python script.")
            python_script = response
        else:
            return None
    if ignore_imports:
        python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE)

    return python_script


def merge_import_dicts(import_dict_1, import_dict_2):
    """

    :param import_dict_1: as returned by code_parser import dict
    :param import_dict_2:
    :return: A SET
    """
    import_dict_merged_set = set(import_dict_1.items()).union(import_dict_2.items())
    return import_dict_merged_set

def create_standard_import(import_dict_merged_set):
    """

    :param import_dict_merged_set: A SET of (k, v): k: full name of imported package OR a tuple (from_path, name); v: as_name


    :return:
    """
    std_import = ""
    for k, v in import_dict_merged_set:
        as_name = v
        if isinstance(k, tuple):
            from_name, name = k

            std_import += f"from {from_name} import {name}"
            if name != as_name and as_name is not None:
                std_import += f" as {as_name}"
        else:  # import
            std_import += f"import {k}"
            if k != as_name and as_name is not None:
                std_import += f" as {as_name}"
        std_import += "\n"

    return std_import