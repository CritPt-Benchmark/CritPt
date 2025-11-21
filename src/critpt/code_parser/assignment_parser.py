import ast
import copy

from critpt.code_parser.name import Name


class AssignmentParser:
    def __init__(self, _text, allow_undefined_variables=False):
        self.text = _text
        self.error_info = None
        self.allow_undefined_variables = allow_undefined_variables
        try:
            self.tree = ast.parse(_text)
        except SyntaxError as e:
            self.tree = None
            self.error_info = f"Syntax error: {e}"

        self._assignment = {}

        try:
            self._dict()
        except AssertionError as e:
            self._assignment = {}
            self.tree = None
            self.error_info = f"{e}"

    def is_valid(self):
        return self.tree is not None

    def get_assignments(self):
        return copy.copy(self._assignment)

    def resolve_rhs_name(self, rhs_value):
        rhs_id = rhs_value.id
        if rhs_id not in self._assignment:
            if self.allow_undefined_variables:
                return Name(rhs_id)
            else:
                raise AssertionError(f"undefined variable {rhs_id}")
        _v = self._assignment[rhs_id]
        return _v

    def unpack(self, rhs_value):
        if isinstance(rhs_value, ast.Name):
            _v = self.resolve_rhs_name(rhs_value)
        else:
            if isinstance(rhs_value, ast.List) or isinstance(rhs_value, ast.Tuple):
                _v = []
                for _e in rhs_value.elts:
                    _v.append(self.unpack(_e))
                if isinstance(rhs_value, ast.Tuple):
                    _v = tuple(_v)
            elif isinstance(rhs_value, ast.Dict):
                _v = {}
                for dict_key, dict_value in zip(rhs_value.keys, rhs_value.values):
                    if dict_key is None:  # requires unpacking the dict_value
                        raise AssertionError(f"Unpacking value {dict_value} in dict is unsupported")
                    assert isinstance(dict_key, ast.Constant), "Key in dict must be a constant"
                    _v[dict_key.value] = self.unpack(dict_value)
            else:
                assert isinstance(rhs_value, ast.Constant), f"element on rhs must be name/list/tuple/dict/constant, but got {rhs_value}"
                _v = rhs_value.value
        return _v


    def _dict(self):
        assert self.tree is not None
        self._assignment = {}
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.Assign):
                for _t in node.targets:
                    assert isinstance(_t, ast.Name), f"unpacking lhs in asssignments unsupported at this time: got {_t}"
                    _v = self.unpack(node.value)
                    self._assignment[_t.id] = _v
