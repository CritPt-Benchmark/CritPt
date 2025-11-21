import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Union

from critpt.code_parser import CodeParser
from critpt import paths
from critpt.notebook_reader.helper_functions import cell_keyword_filter


class NotebookCell:
    """
    Design choice: a header can only be detected at the top of a markdown cell.
    """

    cnt = 0
    def __init__(self, cell_dict):
        self.number = self.cnt
        NotebookCell.cnt += 1
        self.raw_data = cell_dict
        self.cell_type = cell_dict['cell_type']  # "markdown" or "code"
        self.source = []  # a list. Each element is a string that ends with '\n'.
        idx = 0
        for line in cell_dict['source'] :
            if len(self.source) <= idx:
                self.source.append(line)
            else:
                self.source[-1] += line
            if line.endswith("\n"):
                idx += 1
        cell_dict['source'] = self.source
        self.metadata = cell_dict['metadata']
        self._str = ''.join(self.source)
        self.first_nonempty_line_index = -1
        self.first_nonempty_line()
        self.header_level = self._header_level()
        self.title = self._title()
        self.parent = None  # EXTERNAL
        self.children = []  # EXTERNAL

        self.sub_cells = {}
        self.class_name = "cell"
        self._full_class = None  # memoization

        self.other_info = {}


    def split_by_header(self):
        pass

    @classmethod
    def dummy(cls):
        return NotebookCell({"cell_type": "dummy", "source": [], "metadata": None})

    def set_parent(self, parent):
        self.parent = parent
        self._full_class = None

    def add_children(self, *children):
        self.children.extend(children)

    @property
    def full_name(self):
        # if not self._full_class:
        if self.parent:
            self._full_class = self.parent.full_name + "." + self.class_name
        else:
            self._full_class = self.class_name
        return self._full_class

    def __str__(self):
        return self._str

    def to_dict(self):
        # if str(self).lower().find("template") != -1:
        #     print("TO_DICT", str(self), self.full_name)
        return {"string": str(self), "cell_type": self.cell_type, "full_name": self.full_name, "metadata": self.metadata,
                "other_info": self.other_info}

    def first_nonempty_line(self):
        if self.first_nonempty_line_index < 0:
            self.first_nonempty_line_index = 0
            while (self.first_nonempty_line_index < len(self.source) and
                   len(self.source[self.first_nonempty_line_index].strip()) == 0):
                self.first_nonempty_line_index += 1
        if self.first_nonempty_line_index >= len(self.source):
            return None
        return self.source[self.first_nonempty_line_index]

    def is_empty(self):
        if self.first_nonempty_line_index < 0:
            self.first_nonempty_line()
        return len(self.source) == self.first_nonempty_line_index

    @staticmethod
    def is_header(line):
        line = line.strip()
        return re.match(r"^#+\s+", line)

    def _header_level(self):
        """

        :return: 0 if not header; returns an integer that represents the number of # s; -1 means dummy
        """
        if self.cell_type == "markdown" and not self.is_empty():

            first_line = self.first_nonempty_line().strip()
            # print("CELL", self.number, first_line, str(self))
            if first_line.startswith("#"):
                i = 0
                while i < len(first_line):
                    if first_line[i] == "#":
                        i += 1
                        continue
                    if first_line[i].isspace():
                        return i
                    return 0
        if self.cell_type == "dummy":
            return -1
        return 0

    def _title(self):
        h_l = self.header_level
        if h_l > 0:
            return self.first_nonempty_line()[h_l + 1:].strip().rstrip(":")
        return ""

    def get_child(self, idx, cell_type=None):
        raw_idx = idx
        if cell_type is None:
            return self.children[idx]
        else:
            for child in self.children:
                if cell_type == child.cell_type:
                    if idx == 0:
                        return child
                    idx -= 1
        raise ValueError(f"Invalid index: {raw_idx} or cell_type {cell_type}, must be 'markdown' or 'code'")

    def get_sub_cell(self, idx, cell_type=None):
        if idx < len(self.get_sub_cells(cell_type)):
            return self.get_sub_cells(cell_type)[idx]
        return None

    def get_sub_cells(self, cell_type=None):
        if self.sub_cells.get(cell_type) is not None:
            return self.sub_cells[cell_type]
        ret = []
        for cell in self.walk():
            if cell_type is None or cell_type == cell.cell_type:
                ret.append(cell)
        self.sub_cells[cell_type] = ret
        return ret

    def get_sub_header(self, caption, multiple=False) -> Optional[Union['NotebookCell', List['NotebookCell']]]:
        """
        Call after __init__ or titled_cells is constructed

        :param caption:
        :param multiple:
        :return:
        """
        cells = []
        header_filter = cell_keyword_filter("markdown", caption, ignore_case=True)
        for cell in self.walk():
            if header_filter(cell):
                cells.append(cell)
        if multiple:
            return cells
        if len(cells) == 0:
            return None
        return cells[0]

    def walk(self):
        if self.cell_type != "dummy":
            yield self
        for child in self.children:
            yield from child.walk()

    def get_inner_lines(self, with_header=True):
        ret = self.source
        if not with_header and self.header_level > 0:
            ret = ret[1:]
        for child in self.walk():
            ret += child.source
        return ret