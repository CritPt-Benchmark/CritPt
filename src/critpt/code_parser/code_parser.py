import ast
import copy
import re
from pathlib import Path
from typing import List

from critpt.code_parser.helper_functions import extract_python_script, assigned_names


class CodeParser:
    def __init__(self, raw_response, ignore_imports=True):
        if raw_response is None or len(raw_response) == 0:
            self.error_info = ("SyntaxError", "Response is empty")
            self.ast_tree = None
            self.code_str = None
            return
        self.raw_response = raw_response

        self.code_str = extract_python_script(raw_response, ignore_imports=ignore_imports)

        self.error_info = None
        try:
            if self.code_str is None:
                self.ast_tree = None
            else:
                self.ast_tree = ast.parse(self.code_str)
        except SyntaxError as e:
            self.ast_tree = None
            self.code_str = None
            self.error_info = ('SyntaxError', str(e))
            return
        except Exception as e:
            print(f"UNCAUGHT ERROR CONSTRUCTING CODE PARSER: {type(e)}: {e}")
            raise e

        self.function_nodes, self.all_other_nodes = self.get_function_from_code()
        self.function_node, self.function_name, self.function_args, self.function_arg_names = None, None, None, None
        if len(self.function_nodes) == 1:
            self.function_node = self.function_nodes[0]
            self.function_name = self.function_node.name
            self.function_args = self.function_node.args
            self.function_arg_names = {
                "pos_only_args": [arg.arg for arg in self.function_args.posonlyargs],
                "args": [arg.arg for arg in self.function_args.args],
                "var_args": self.function_args.vararg.arg if self.function_args.vararg else None,
                "kw_only_args": [arg.arg for arg in self.function_args.kwonlyargs],
                "kwarg": self.function_args.kwarg.arg if self.function_args.kwarg else None
            }
            self.function_arg_names["fixed_params"] = (self.function_arg_names["pos_only_args"] +
                                                       self.function_arg_names["args"] +
                                                       self.function_arg_names["kw_only_args"])
        else:
            self.error_info = ('FormatError', f'Code parser currently only accept code block with exactly one function. Received {len(self.function_nodes)} functions: \n{self.code_str}')


    def get_import_dict(self, return_anything_else=False):
        """

        :return: a dict, {original: alias} where original is the full name of the import separated by '.' and alias is the part after as
        """
        # print(self.ast_tree)
        # print(self.error_info)
        ret_dict = {}
        tree = copy.deepcopy(self.ast_tree)
        tree.body = []
        for node in ast.iter_child_nodes(self.ast_tree):
            # print(node)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    ret_dict[(node.module, alias.name)] = alias.asname
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    ret_dict[alias.name] = alias.asname
            else:
                tree.body.append(node)

        if return_anything_else:
            return ret_dict, ast.unparse(tree)
        return ret_dict

    def get_variable_names_from_ast(self, root=None):
        all_names = set()
        if root is None:
            root = self.ast_tree
        for node in ast.iter_child_nodes(root):
            if names := assigned_names(node):
                all_names.update(names)
        return list(all_names)

    def get_function_from_code(self):
        """
        Extracts and returns the source code of the specified function from a given source code string.

        :param function_name: Name of the function to extract
        :return: String containing the source code of the function, or None if the function is not found; and all other code
        """

        try:
            # Iterate through all nodes in the AST
            function = []
            all_others = []

            for node in ast.iter_child_nodes(self.ast_tree):
                # Check if the node is a function definition
                if isinstance(node, ast.FunctionDef):
                    # Convert the AST back to a string containing the Python code for the function
                    function.append(node)
                else:
                    all_others.append(node)
            return function, all_others
        except Exception as e:
            print(f'Function not found with error: {e}')
            return [], [self.ast_tree]

    def get_function_str(self):
        return '\n'.join([ast.unparse(node) for node in self.function_nodes])


    # create arg lists for calling
    def create_call_str(self, target_parser=None):
        call_list = self.create_call_args(target_parser)
        call_str = ', '.join([f'{arg}={arg}' for arg in call_list])

        return call_str

    def create_call_args(self, target_parser=None, variable_list=None):
        if target_parser is None:
            assert variable_list is not None
        else:
            variable_list = target_parser.get_variable_names_from_ast()
        arg_list = self.function_arg_names["fixed_params"] or []
        call_list = [arg for arg in arg_list]

        return call_list



if __name__ == '__main__':
    path = Path(r'D:\Documents\Other_Projects\cortexInspect\results\WF_SYK_entropy_v0p\use_golden_for_prev_steps_False\parsing_False\deepseek\deepseek-reasoner\sub_1.parsed.out')
    with open(path, 'r', encoding='utf-8') as f:
        source_code = f.read()
        parser = CodeParser(source_code)
        print(parser.get_function_str())
