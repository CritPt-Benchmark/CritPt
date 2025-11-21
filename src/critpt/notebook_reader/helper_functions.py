import re
from typing import List, Callable, Optional


# from critpt.notebook_reader.notebook_cell import NotebookCell


def get_indent(line: str):
    return ' ' * (len(line) - len(line.lstrip(' ')))


def fill_code_template(template_cell: Optional['NotebookCell'], filler_cell: Optional['NotebookCell']):
    if template_cell is None:
        return None
    if filler_cell is None:
        return str(template_cell)
    template_fill_positions = []  # (line_idx, variable_name or None)
    default_template_fill_position = None  # insert before this position

    template_list = template_cell.source

    function_idx = -1
    new_function_name = ""
    others = ")"
    for idx, template_cell_line in enumerate(template_list):
        line = template_cell_line.split("#", 1)[0].strip()
        if line.endswith("..."):  # indicating a template fill position
            line_assignment = line.split("=", 1)
            if len(line_assignment) == 2:
                line_assignment = line_assignment[0].strip()
            else:
                line_assignment = None
            template_fill_positions.append((idx, line_assignment))
        if line.startswith("return"):
            default_template_fill_position = idx
        if line.startswith("def"):
            function_name, others = line[3:].strip().split("(", 1)
            new_function_name = "real_answer"
            function_idx = idx
        if function_idx != -1:
            template_list[function_idx] = "def " + new_function_name + "(" + others + "\n"

    if len(template_fill_positions) == 0 and default_template_fill_position is None:
        return "".join([template_list, filler_cell.source])

    if len(template_fill_positions) > 0:
        default_template_fill_position = template_fill_positions[0][0]
        indent = get_indent(template_list[default_template_fill_position])
        template_list = template_list[:template_fill_positions[0][0]] + template_list[
                                                                        (template_fill_positions[-1][0] + 1):]
    else:
        indent = get_indent(template_list[default_template_fill_position])
    return "".join(template_list[:default_template_fill_position] +
                   [indent + line.strip() + "\n" for line in filler_cell.source] +
                   template_list[default_template_fill_position:])


def cell_keyword_filter(
    cell_type: str | None,
    *keywords: str,
    ignore_case: bool = True
) -> Callable[['NotebookCell'], bool]:
    """

    :param cell_type: None or "code" or "markdown"
    :param keywords: a list of keywords that must be present in the cell
    :param ignore_case:
    :return: Returns a filter function that takes a cell and returns True only if all keywords are present in that cell
    """

    def _filter(cell: 'NotebookCell') -> bool:
        if cell_type is not None and cell.cell_type != cell_type:
            return False
        if not keywords:
            return True
        for keyword in keywords:
            header_level = 0
            if keyword.startswith("#"):
                marker, keyword = keyword.split(" ", 1)
                header_level = marker.count("#")
                if marker.endswith("*"):
                    header_level = -1
            if header_level == -1 or cell.header_level == header_level:
                if re.fullmatch(keyword, cell.title, flags=re.IGNORECASE if ignore_case else 0):
                    continue
            if header_level == 0:  # not a header keyword
                if re.match(keyword, str(cell), flags=re.IGNORECASE if ignore_case else 0):
                    continue
            return False
        return True

    return _filter


def cell_hierarchy_filter(
    name: str | List[str] | None = None,
    exclude: str | List[str] | None = None,
    ignore_case: bool = True,
    flexible: bool = True
) -> Callable[['NotebookCell'], bool]:
    """
    Returns a filter function that compares the full name (hierarchy) of the cell to the provided filter args.
    :param name: None or str or List(str). The cell must include all names to pass.
    :param exclude: None or str or List(str). The cell must include none of the excluded names to pass.
    :param ignore_case:
    :param flexible: hierarchy (name and exclude) is given by a string with multiple names separated by ".". If this is
        true, then a name after "." is a descendant (not limited to a child) of that before the same ".".
    :return:
    """

    def _filter(cell: 'NotebookCell') -> bool:
        _keyword = name
        _exclude = exclude
        cell_name = cell.full_name

        def to_lower(thing):
            if isinstance(thing, str):
                return thing.lower()
            if isinstance(thing, list):
                return [to_lower(item) for item in thing]
            return thing

        if ignore_case:
            _keyword = to_lower(_keyword)
            _exclude = to_lower(_exclude)
            cell_name = to_lower(cell_name)

        def find_hierarchy(full_name, hierarchy):
            """
            Greedy method of finding hierarchy.
            :param full_name:
            :param hierarchy:
            :return:
            """
            if flexible:
                names = hierarchy.split(".")
            else:
                names = [hierarchy]

            src_hierarchy = full_name.split(".")
            for _name in names:
                found_flag = False
                while len(src_hierarchy) > 0:
                    src_name = src_hierarchy.pop(0)
                    if src_name == _name:
                        found_flag = True
                        break
                if not found_flag:
                    return False

            return True


        # keyword judge
        if _keyword is not None:
            if isinstance(_keyword, str):
                _keyword = [_keyword]
            for kw in _keyword:
                if not find_hierarchy(cell_name, kw):
                    return False

        # exclude judge
        if _exclude is not None:
            if isinstance(_exclude, str):
                _exclude = [_exclude]
            for ex in _exclude:
                if find_hierarchy(cell_name, ex):
                    return False

        return True

    return _filter

def filter_or(*filters: Callable[['NotebookCell'], bool]) -> Callable[['NotebookCell'], bool]:
    def _filter(cell: 'NotebookCell') -> bool:
        for cell_filter in filters:
            if cell_filter(cell):
                return True
        return False
    return _filter


def set_class_name(_cell: Optional['NotebookCell'], class_name: str) -> None:
    if _cell:
        _cell.class_name = class_name
