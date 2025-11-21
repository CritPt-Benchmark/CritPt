import ast
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Union

from critpt.code_parser import CodeParser
from critpt import paths
from critpt.code_parser.assignment_parser import AssignmentParser
from critpt.notebook_reader.helper_functions import set_class_name, fill_code_template, cell_keyword_filter, \
    cell_hierarchy_filter, filter_or

from critpt.notebook_reader.notebook_cell import NotebookCell

class Reader:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.cells: List[NotebookCell] | None = None
        self.code_cells = []
        self.md_cells = []
        self.dummy = None
        self.get_cells()

        self.titled_cells = []
        for i, cell in enumerate(self.md_cells):
            if cell.header_level > 0:
                self.titled_cells.append({"cell": cell, "title": cell.title, "header_level": cell.header_level})

        self.tag = self.header("# Tag")
        self.contains_error = False
        # detect main problem
        self.problem_setup = self.header("# Problem setup")
        self.main_problem = self.header("# Main problem")

        self.main_problem_code_template, self.main_problem_answer_only_code, self.main_problem_answer_code,\
            self.main_testcases = (
            self.get_problem_code(self.main_problem))

        # detect sub problems
        self.sub_problem_header = self.header("# Subproblems")
        _sub_problems: List[NotebookCell] = self.header("#* Subproblem\\s+\\d+", multiple=True)

        # filter
        _new_sub_problems = {}
        for sub_problem in _sub_problems:
            normalized_title = sub_problem.title.replace("#", '').replace(":", '').lower().strip()
            if normalized_title not in _new_sub_problems:
                _new_sub_problems[normalized_title] = sub_problem

        self.sub_problems = list(_new_sub_problems.values())

        ___ = [self.get_problem_code(sub_problem)
                                                       for sub_problem in self.sub_problems]

        self.sub_problem_code_templates, self.sub_problem_answer_only_codes, \
            self.sub_problem_answer_codes, self.sub_testcases = list(zip(*___)) if len(___) else ([], [], [], [])

        # get lit. answers
        self.lit_answers = self.header("#* Answer", multiple=True)



        set_class_name(self.tag, "tag")
        set_class_name(self.problem_setup, "setup")
        set_class_name(self.main_problem, "main")
        set_class_name(self.main_problem_code_template, "template")
        # print(",,,m", self.main_problem_code_template.parent, self.main_problem_code_template.parent.full_name)
        # sub_code_template = self.sub_problem_code_templates[0]
        # print(",,,", sub_code_template.parent, sub_code_template.parent.full_name,
        #       sub_code_template.parent.class_name)
        set_class_name(self.main_problem_answer_only_code, "answer_code")
        set_class_name(self.sub_problem_header, "sub_header")
        for sub_problem, sub_code_template, sub_ans_only_code in zip(self.sub_problems, self.sub_problem_code_templates, self.sub_problem_answer_only_codes):
            # print(",,,", sub_code_template.parent, sub_code_template.parent.full_name,
            #       sub_code_template.parent.class_name)
            set_class_name(sub_problem, "sub")
            set_class_name(sub_code_template, "template")
            sub_problem.other_info.update({"parsed": sub_code_template is not None})

            set_class_name(sub_ans_only_code, "answer_code")
            # print(",,,", sub_code_template.parent, sub_code_template.parent.full_name, sub_code_template.parent.class_name)
        for answer in self.lit_answers:
            set_class_name(answer, "answer")

        # for cell in self.cells:
            # print(cell.full_name, cell.number)  # ???



    def get_problem_code(self, problem_cell: NotebookCell) -> Tuple[
        NotebookCell | None, NotebookCell | None, str | None, list | None]:
        if problem_cell is None:
            return None, None, "", None
        code_template_header = self.header(r'#* Parsing template', root=problem_cell)
        answer_code_header = self.header(r'#* Answer\scode', root=problem_cell)

        if code_template_header:
            set_class_name(code_template_header, 'template_header')
        if answer_code_header:
            set_class_name(answer_code_header, 'answer_code_header')

        if code_template_header:
            code_template = code_template_header.get_sub_cell(0, 'code')
        else:  # LEGACY
            code_template = problem_cell.get_sub_cell(0, "code")

        if answer_code_header:
            answer_only_code = answer_code_header.get_sub_cell(0, "code")
        else:  # LEGACY
            answer_only_code = problem_cell.get_sub_cell(1, "code")

        self.get_settings(problem_cell.title, problem_cell)

        # get test cases
        tc = problem_cell.get_sub_header("#* Test\s*cases")
        print("TEST CASES", tc)
        set_class_name(tc, "testcases")
        if tc is None:
            tc_list = None
        else:
            tc_string = tc.get_sub_cell(0, "code").__str__()
            # construct testcase object which should be a list of dicts
            ap = AssignmentParser("testcases = " + tc_string, allow_undefined_variables=True)

            if ap.error_info is not None:
                raise RuntimeError(ap.error_info)
            tc_list = ap.text  # ap.get_assignments()["tc"]

        filled_code_template = fill_code_template(code_template, answer_only_code)

        return code_template, answer_only_code, filled_code_template, tc_list

    def get_problem_cell_from_step(self, step):
        """

        :param step: int or "main"
        :return: a cell
        """
        if step == 'main':
            cell = self.main_problem
        else:
            cell = self.sub_problems[step]
        return cell

    def get_settings(self, step, cell=None):
        """

        :param step: int (sub) or "main" or None (only if cell is set)
        :param cell: optional; THE cell of the step. If this is passed, it is safe to set step to None.
        :return:
        """
        if cell is None:
            cell = self.get_problem_cell_from_step(step)
        settings_cell = cell.get_sub_header('#* Settings')
        set_class_name(settings_cell, 'error')
        error_cell = cell.get_sub_header('#* Error')
        set_class_name(error_cell, 'error')
        if not settings_cell:  # read from settings cell; if there is no settings cell refer to error cell
            settings_cell = error_cell
        settings_dict = {}
        if settings_cell:
            assignment_contents = ""
            for line in settings_cell.get_inner_lines(with_header=False):  # if there is a code block, read from it too

                if line.find("=") != -1:
                    assignment_contents += line + "\n"
                    _a_parser = AssignmentParser(assignment_contents)
                    assert _a_parser.is_valid(), f"Error parsing Settings in file {self.path}, {step}: \n>    {line}\nError info: {_a_parser.error_info}"
                    assignments = _a_parser.get_assignments()
                    settings_dict.update(assignments)
        if 'error' not in settings_dict:
            error = self.legacy_get_error(step, cell)
            if error is not None:
                settings_dict['error'] = error
        else:
            settings_dict['error'] = settings_dict['error']
        if len(settings_dict):
            self.contains_error = True
        return settings_dict

    def legacy_get_error(self, step, cell=None):
        if cell is None:
            cell = self.get_problem_cell_from_step(step)
        error_cell = cell.get_sub_header('#* Error')
        if error_cell:
            for line in error_cell.get_inner_lines(with_header=False):
                number_re_pattern = r'-?(?:(?:[1-9]\d*|0)(?:\.\d*)?|\.\d+)'
                if re.fullmatch(number_re_pattern, line.strip()):
                    return float(line)

    def get_cells(self):
        if self.cells is None:
            self.cells = [NotebookCell(cell)
                          for cell in self.raw_data['cells']]

            self.md_cells = [cell for cell in self.cells if cell.cell_type == "markdown"]
            self.code_cells = [cell for cell in self.cells if cell.cell_type == "code"]
            self.dummy = NotebookCell.dummy()

            last_cell = self.dummy
            for cell in self.cells:

                while last_cell and (
                        last_cell.header_level == 0 or (cell.header_level != 0 and last_cell.header_level >= cell.header_level)):
                    # print("LAST", last_cell, last_cell.header_level)
                    # print("THIS", cell, cell.header_level)
                    last_cell = last_cell.parent

                if last_cell:
                    last_cell.add_children(cell)
                cell.set_parent(last_cell)
                last_cell = cell
                # print("------------------")
                # print(cell)
                # print("HEADER_LEVEL", cell.header_level)
                # print("PARENT =", cell.parent.source[:2])
                # print("PARENT HEADER LV =", cell.parent.header_level)

        return self.cells

    def header(self, caption, multiple=False, root=None) -> NotebookCell | List[NotebookCell] | None:
        """
        Call after __init__ or titled_cells is constructed

        :param caption:
        :param multiple:
        :return:
        """
        cells = []
        header_filter = cell_keyword_filter("markdown", caption, ignore_case=True)
        if root is not None:
            for cell in root.walk():
                if header_filter(cell):
                    cells.append(cell)
        else:
            for cell_info in self.titled_cells:
                cell = cell_info["cell"]
                if header_filter(cell):
                    cells.append(cell)
        if multiple:
            return cells
        if len(cells) == 0:
            return None
        return cells[0]

    # def get_main_problem_prompt(self):
    #     return (str(self.problem_setup) + "\n\n" +
    #             str(self.main_problem) + "\n\n```python\n" +
    #             str(self.main_problem_code_template) + "\n```")
    #
    # def get_sub_problem_prompt(self, until_step=None):
    #     return "\n\n".join(  # [self.get_main_problem_prompt(), (str(self.sub_problem_header) or "")] +
    #         [
    #             '\n\n'.join([str(problem), "```python\n" + str(
    #                 problem_code_template) + "\n```" if problem_code_template is not None else '(Skipped as subproblem code is incomplete)'])
    #             for problem, problem_code_template in
    #             zip(self.sub_problems[:until_step], self.sub_problem_code_templates[:until_step])
    #         ])

    def to_list(self, cell_filter=None):
        return [cell.to_dict() for cell in self.cells if cell_filter is None or cell_filter(cell)]

    def get_main_problem_list(self):
        cell_filter = filter_or(cell_hierarchy_filter("setup"), cell_hierarchy_filter("main"))
        return self.to_list(cell_filter)

    def get_sub_problem_list(self):
        cell_filter = filter_or(cell_hierarchy_filter("setup"), cell_hierarchy_filter(exclude="main"))
        return self.to_list(cell_filter)

    @classmethod
    def get_main_problem_prompt(cls, problem_list, parsing=True):
        """
        This method is such defined that with only a full list of cells you can determine the prompt
        :param problem_list:
        :return:
        """
        problem_string_list = []
        for cell_dict in problem_list:
            if cell_dict["full_name"].find("main") == -1 and cell_dict["full_name"].find("setup") == -1:
                continue
            if cell_dict["full_name"].find("answer") != -1:
                continue
            if cell_dict["full_name"].find("template") != -1 and not parsing:
                continue
            if cell_dict["full_name"].find("error") != -1:
                continue
            if cell_dict["full_name"].find("testcases") != -1:
                continue
            if cell_dict["cell_type"] == "code":
                problem_string_list.append("```python\n" + cell_dict["string"] + "\n```")
            else:
                problem_string_list.append(cell_dict["string"])
        return "\n\n".join(problem_string_list)

    @classmethod
    def get_sub_problem_prompt(cls, problem_list, from_step=0, until_step=None, parsing=True,
                               skip_unparsed=True, multiturn_with_answer=False):
        """

        :param problem_list:
        :param from_step: [from_step, until_step)
        :param until_step: [from_step, until_step)
        :return:
        """
        problem_string_list = []
        current_step = -1

        if until_step is None:

            num_steps = 0
            for cell_dict in problem_list:
                if cell_dict["full_name"].endswith("sub"):
                    num_steps += 1
            until_step = num_steps

        if from_step is None:
            from_step = 0

        for cell_dict in problem_list:
            if from_step and cell_dict["full_name"].find("setup") != -1:

                continue  # if not starting from the beginning then do not contain setup
            if cell_dict["full_name"].find("sub") == -1 and cell_dict["full_name"].find("setup") == -1:
                continue
            if cell_dict["full_name"].endswith("sub"):  # the header cell of a sub problem
                current_step += 1
            if cell_dict["full_name"].find("sub") != -1 and (current_step < from_step or until_step <= current_step):
                # previous answer
                if not multiturn_with_answer or current_step != until_step - 2:
                    continue

            if cell_dict["full_name"].find("template") != -1 and not parsing:
                continue
            if cell_dict["full_name"].find("error") != -1:
                continue
            if cell_dict["full_name"].find("testcases") != -1:
                continue

            # previous answer
            if multiturn_with_answer and current_step == until_step - 2:
                if cell_dict["full_name"].endswith(".answer"):
                    lines = cell_dict["string"].split("\n")
                    answer_str = "\nThe correct answer is:\n"
                    for line in lines:
                        if line.find("# Answer") == -1:
                            answer_str += line + "\n"
                    answer_str += "\nNow let's solve the next Checkpoint.\n"
                    problem_string_list.append(answer_str)
                if current_step > -1: # we should include problem setup before sub 1
                    continue


            # if not the last step, ignore template and code answer
            elif current_step != until_step - 1:
                if cell_dict["full_name"].find("template") != -1 or cell_dict["full_name"].find("code") != -1:
                    continue
            # else ignore answer
            else:
                if cell_dict["full_name"].find("answer") != -1:
                    continue

            if cell_dict["cell_type"] == "code":
                problem_string_list.append("```python\n" + cell_dict["string"] + "\n```")
            else:
                problem_string_list.append(cell_dict["string"])
            # print(">>>>", cell_dict['full_name'], "@@@", cell_dict['string'][:100], '@@@@', )
        return "\n\n".join(problem_string_list)

    def is_main_problem_parsed(self):
        return self.main_problem_code_template is not None

    def is_sub_problem_parsed(self, step):
        return self.sub_problem_code_templates[step] is not None

    def parsed_sub_problems(self):
        """
        :return: a list of int, steps of all parsed sub problems.
        """

        steps = [step for step in range(len(self.sub_problems)) if self.is_sub_problem_parsed(step)]
        return steps

