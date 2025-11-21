import copy
import re
from collections import OrderedDict
from os import PathLike
from pathlib import Path

import yaml
from jinja2 import Environment, meta, Template, Undefined
from critpt.paths import TEMPLATES_ROOT

SYSTEM_ROOT_VAR = "SYSTEM_PROMPT"
PARSE_ROOT_VAR = "PARSE_PROMPT"
DEFAULT_TEMPLATE_PATH = TEMPLATES_ROOT / "prompt_template_default.yaml"
DEFAULT_TEMPLATE_ONE_STEP_PATH = TEMPLATES_ROOT / "prompt_template_default_one_step.yaml"
DEFAULT_TEMPLATE_TWO_STEP_PATH = TEMPLATES_ROOT / "prompt_template_default_two_step.yaml"

LEAF_VAR_PREFIX = "PROMPT_SPECS_"
IMPORTS_VAR = "IMPORTS"

_LAZY_INIT_DEFAULT_SYSTEM_PROMPT = None  # lazy
_LAZY_INIT_DEFAULT_PARSE_PROMPT = None


class PreserveUndefined(Undefined):
    def __str__(self):
        # Reproduce the original Jinja syntax, including filters
        if self._undefined_hint is not None:
            return self._undefined_hint
        return f"{{{{ {self._undefined_name} }}}}"

    def _fail_with_undefined_error(self, *args, **kwargs):
        return str(self)

    def __iter__(self):  # allow iteration over undefined
        return iter(())

    def __getitem__(self, key):  # preserve dict-like access
        return f"{{{{ {self._undefined_name}[{repr(key)}] }}}}"

    def __getattr__(self, attr):  # ~~preserve attribute access~~
        raise AttributeError(f"'PreserveUndefined' object has no attribute '{attr}'")
        # return f"{{{{ {self._undefined_name}.{attr} }}}}"

    def __call__(self, *args, **kwargs):  # preserve call
        return f"{{{{ {self._undefined_name}() }}}}"


class BaseTemplate:
    _class_level_template_dict_records = {}

    @staticmethod
    def read_template(template_path: Path | None = None,
                      template_dict: dict | None = None,
                      imports_var: str | None = IMPORTS_VAR) -> dict:

        if template_dict is None:
            # Read with memoization: one template file is only opened once
            if template_path not in BaseTemplate._class_level_template_dict_records:
                with open(template_path, "r", encoding="utf-8") as f:
                    template_dict = yaml.safe_load(f)

            else:
                return BaseTemplate._class_level_template_dict_records[template_path]

        _imports = template_dict.get(imports_var, [])
        if isinstance(_imports, str):
            _imports = [_imports]

        base_dict = {}

        # preprocess _imports
        import_format_dict = """
        When using dict-style import, provide paths like this:
        IMPORTS:
          template:
          - xxx.html
          - yyy.html
          - zzz.html
          KEY: www.yaml                    <Note: contents of www.yaml will be provided as the value of KEY>
        """
        import_format_list = """
        When using list-style import, provide paths like this:
        IMPORTS:
          - xxx.yaml                       <Note: parsed as a template by default>
          -   template: 
              - yyy.yaml
          -   KEY1: zzz.html
          -   template: www.yaml
              KEY2: sss.html
        """
        _temp_imports = []
        for item in _imports:
            if isinstance(_imports, dict):

                paths = _imports[item]
                if isinstance(paths, str):
                    paths = [paths]

                assert isinstance(paths, list), f"Wrong {item} import:" + import_format_dict
                if item != "template":
                    assert len(paths) == 1, f"Wrong {item} import:" + import_format_dict

                for path in paths:
                    _temp_imports.append([item, path])

            else:  # isinstance(_imports, list)
                assert isinstance(_imports, list), (
                    "Import should be either a string, a dict, or a list." +
                    import_format_dict + import_format_list
                )

                if isinstance(item, dict):
                    for key in item:
                        paths = item[key]
                        if isinstance(paths, str):
                            paths = [paths]
                        assert isinstance(paths, list), f"Wrong {key} import:" + import_format_list
                        if item != "template":
                            assert len(paths) == 1, f"Wrong {item} import:" + import_format_dict
                        for path in paths:
                            _temp_imports.append([key, path])
                else:  # isinstance(item, str)
                    assert isinstance(item, str), f"Wrong template import:" + import_format_list
                    _temp_imports.append(["template", item])


        _imports = _temp_imports

        # recursive import
        for style, import_name in _imports:
            # path expansion rule:
            # - if the current template is from a file, then all relative paths are based on the directory
            #   that holds the current template file
            # - if the current template is given as a dict, then all relative paths are based on the working directory


            if template_path is not None:
                base_path = template_path.parent
                import_path = base_path / import_name
            else:
                import_path = import_name
            import_path = import_path.resolve(strict=True)

            if style == "template":
                # Import order matters: later imports will be prioritized over earlier ones
                base_dict.update(BaseTemplate.read_template(import_path, None, imports_var))

            else:
                with open(import_path, "r", encoding="utf-8") as f:
                    base_dict.update({style: f.read()})


        # overwrite base_dict with template_dict
        base_dict.update(template_dict)
        BaseTemplate._class_level_template_dict_records[template_path] = base_dict

        return base_dict

    def __init__(self, root_var,
                 leaf_var_prefix=LEAF_VAR_PREFIX,
                 imports_var=IMPORTS_VAR,
                 template_path: str | PathLike[str] = None, template_dict: dict = None,
                render=True,
                 **other_template_vars):
        f"""
        
        :param root_var: a variable to start parsing with.
        :param template_path: a yaml file that contains template variables. 'prompt_template.yaml' by default
        :param template_dict: a dictionary that contains template variables. Only one of template_path 
            or template_dict should be set.
        :param other_template_vars: Updates template_dict with new variables and overwrite existing variables 
        """
        assert template_path is None or template_dict is None
        self.root_var = root_var
        self.leaf_var_prefix = leaf_var_prefix
        if template_path is None and template_dict is None:
            # print("TEMPLATE_PATH", template_path)
            template_path = DEFAULT_TEMPLATE_PATH
        # print("TEMPLATE PATH", template_path)
        self.template_dict = self.read_template(template_path, template_dict, imports_var)

        self.template_dict.update(other_template_vars)
        self.env = Environment(undefined=PreserveUndefined)
        self.rendered_dict = {}
        self.partially_rendered_dict = {}

        self._nonrecursive_templates = {}

        if render:

            self.expanded_template_str = self._renderer(self.root_var, True)
            self.expanded_template = self.env.from_string(self.expanded_template_str)
            self.rendered_template_str = self._renderer(self.root_var, False)
        else:
            self.expanded_template_str = ""
            self.rendered_template_str = None
            self.expanded_template = None
        # print(self.expanded_template_str)

    def render(self, *, root=None, recursive=True, **template_vars):
        """

        :param template_vars: Must be leaf vars.
        :return:
        """

        if root is None:
            root = self.root_var

        if len(template_vars) == 0 and self.rendered_template_str is not None:
            return self.rendered_template_str



        template_dict = copy.copy(self.template_dict)
        template_dict.update(template_vars)  # substitute the vars
        # print("_TEMPLATE_VARS", template_vars)
        # print("TEMPLATE DICT", template_dict)
        x = self._renderer(root, _partially_rendered_dict={},
                           _rendered_dict=(_rendered_dict := {}), _template_dict=template_dict,
                           partial=False, recursive=recursive)
        # print(_rendered_dict)
        return x

    def _renderer(self, var, partial=True, _stack=None,
                  _rendered_dict=None, _partially_rendered_dict=None,
                  _template_dict=None, recursive=True):
        """

        :param var:
        :param partial: ignore leaf vars
        :param _stack:
        :return:
        """

        if _rendered_dict is None:
            _rendered_dict = self.rendered_dict

        if _partially_rendered_dict is None:
            _partially_rendered_dict = self.partially_rendered_dict

        if _template_dict is None:
            _template_dict = self.template_dict

        if partial:
            if var in _partially_rendered_dict:
                return _partially_rendered_dict[var]
            if var in _rendered_dict:  # a leaf
                return var  # as-is
        else:
            if var in _rendered_dict:
                return _rendered_dict[var]

        if var not in _template_dict:
            if var.startswith(self.leaf_var_prefix) and partial is True:  # is a leaf that is not provided
                return var
            return var
            # raise KeyError(f"Variable {var} in template is not provided.")

        if _stack is None:
            _stack = OrderedDict()  # preserve insertion order

        _stack[var] = None

        # first find all dependencies and render
        # print("VAR", var)
        # print(_template_dict[var])
        ast = self.env.parse(_template_dict[var])
        refs = meta.find_undeclared_variables(ast)
        for ref in refs:
            if ref in _stack:
                raise KeyError(f"Detected recursive reference in template: " +
                               '->'.join(k for k in _stack.keys()) +
                               f"[->{ref}]")
            if recursive:
                self._renderer(ref, partial=partial, _stack=_stack, _rendered_dict=_rendered_dict,
                               _partially_rendered_dict=_partially_rendered_dict,
                               _template_dict=_template_dict, )

        # check if current is a leaf
        is_leaf = var.startswith(self.leaf_var_prefix) and len(refs) == 0

        # render current var
        # first preserve if blocks
        repld = False

        def repl(m, known_vars):
            """
            Preserves blocks in m if there is an undefined variable
            :param m: regexp match. There should be a capturing group for ``condition'' where the undefined variables are checked
            :return:
            """
            condition = m.group(1)
            block = m.group(0)

            quoted_pattern = """('''[\\s\\S]*?'''   | \"""[\\s\\S]*?\"""   |   '[^'\\\\]*(?:\\.[^'\\\\]*)*' | "[^"\\\\]*(?:\\.[^"\\\\]*)*"  )"""

            quoted_spans = [w.span() for w in re.finditer(quoted_pattern, condition, re.VERBOSE)]
            # print("CONDITION", condition)
            # print("QUOTED", quoted_spans)
            unquoted_regions = []
            prev_end = 0
            for start, end in quoted_spans:
                if prev_end < start:
                    unquoted_regions.append(condition[prev_end:start])
                prev_end = end
            if prev_end < len(condition):
                unquoted_regions.append(condition[prev_end:])
            unquoted = ' '.join(unquoted_regions)

            # Heuristic: if condition uses unknown vars, return raw block

            vars_in_cond = set(re.findall(r'\b\w+\b', unquoted))
            if not vars_in_cond.issubset(known_vars):
                nonlocal repld
                repld = True
                # escape things
                block = block.replace("\\", "\\\\").replace("\"", "\\\"")
                # print("!", f'{{{{"{block}"}}}}')
                return f'{{{{"{block}"}}}}'
            return block  # keep original for known vars

        if_pattern = re.compile(r'{%\s*(?:if|for) (.*?)\s*%}.*?{%\s*end(?:if|for)\s*%}', re.DOTALL)
        template_string = str(_template_dict[var])
        # print("TEMPLATE STRING", template_string)
        preserved_string_in_full_rendering = if_pattern.sub(lambda m: repl(m, _template_dict.keys()), template_string)
        preserved_string_in_partial_rendering = if_pattern.sub(
            lambda m: repl(m, _partially_rendered_dict.keys()),
            template_string
        )  # All children with a key present in template_dict should now have been registered if they are not leaves


        # print("PRESERVED STRING", preserved_string_in_full_rendering)
        template = self.env.from_string(preserved_string_in_full_rendering)
        partial_template = self.env.from_string(preserved_string_in_partial_rendering)
        partially_rendered = partial_template.render(_partially_rendered_dict)
        rendered = template.render(_rendered_dict)
        # if repld:
            # print("REPL RENDER RESULT:\n  ", rendered)
        if is_leaf:  # leaves are not stored in partially_rendered_dict
            _rendered_dict[var] = partially_rendered
        else:
            _partially_rendered_dict[var] = partially_rendered
            _rendered_dict[var] = rendered

        _stack.pop(var)

        if partial:
            return partially_rendered
        return rendered


class SystemPrompt(BaseTemplate):
    def __init__(self, root_var=SYSTEM_ROOT_VAR,
                 leaf_var_prefix=LEAF_VAR_PREFIX,
                 imports_var: str = None,
                 template_path: str | PathLike[str] = None, template_dict: dict = None,
                 prompt_specs_style=None,
                 **other_template_vars):
        if prompt_specs_style is not None:
            other_template_vars["PROMPT_SPECS_STYLE"] = prompt_specs_style
            if template_dict is None and template_path is None:
                template_path = DEFAULT_TEMPLATE_PATH
        # print("TEMPLATE PATH", template_path)
        super().__init__(root_var, leaf_var_prefix, imports_var, template_path, template_dict, **other_template_vars)

    @classmethod
    def default_system_prompt(cls, prompt_specs_style):
        """

        :param prompt_specs_style: "one-step" or "two-step"
        :return:
        """
        global _LAZY_INIT_DEFAULT_SYSTEM_PROMPT
        if _LAZY_INIT_DEFAULT_SYSTEM_PROMPT is None:

            _LAZY_INIT_DEFAULT_SYSTEM_PROMPT = SystemPrompt(prompt_specs_style=prompt_specs_style)
        return _LAZY_INIT_DEFAULT_SYSTEM_PROMPT.render(PROMPT_SPECS_STYLE=prompt_specs_style)


class ParsePrompt(BaseTemplate):
    def __init__(self, root_var=PARSE_ROOT_VAR,
                 leaf_var_prefix=LEAF_VAR_PREFIX,
                 imports_var: str = None,
                 template_path: str | PathLike[str] = None, template_dict: dict = None,
                 **other_template_vars):
        super().__init__(root_var, leaf_var_prefix, imports_var, template_path, template_dict, **other_template_vars)

    @classmethod
    def default_system_prompt(cls, code_template=None):
        global _LAZY_INIT_DEFAULT_PARSE_PROMPT
        if _LAZY_INIT_DEFAULT_PARSE_PROMPT is None:
            _LAZY_INIT_DEFAULT_PARSE_PROMPT = ParsePrompt()
        kwargs = {}
        if code_template:
            kwargs.update({"PROMPT_SPECS_ANSWER_CODE_TEMPLATE": "{% raw %}" + code_template + "{% endraw %}"})
        return _LAZY_INIT_DEFAULT_PARSE_PROMPT.render(**kwargs)


def simple_test():
    prompt = SystemPrompt(template_path=TEMPLATES_ROOT / Path('./prompt_template_example.yaml'))
    print(prompt.render())
    print()
    print(prompt.render(FLAG=True))
    print()
    print(prompt.render(FLAG=False))


if __name__ == '__main__':
    prompt = SystemPrompt()
    print("@0")
    print(prompt.render())
    print()
    print("@1")
    print(prompt.expanded_template_str)
    print()
    print("@2")
    print(prompt.render(PROMPT_SPECS_PRECISION_DECIMAL=10))
    print()
    print("@3")
    print(prompt.render(PROMPT_SPECS_STYLE='two-step'))
    print()
    print("@4")
    print(SystemPrompt.default_system_prompt("two-step"))
