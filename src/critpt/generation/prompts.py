"""
Prompt helpers for generation.

Generation only needs three components per the new, simplified flow:
1. Assemble the system prompt.
2. Load the problem text into a user prompt.
3. Optionally attach a template parse prompt to extract code.

This module keeps those responsibilities small and composable so the solver can
focus on orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from critpt.templates import get_default_system_prompt


@dataclass
class StepSpec:
    name: str
    prompt: str
    code_template: str
    problem_id: Optional[str] = None


def build_system_prompt(parsing: bool, use_python: bool = False, use_web_search: bool = False) -> str:
    """Return the fully-populated system prompt for the current configuration."""
    system_prompt = get_default_system_prompt(parsing)

    if use_python:
        needle = "Justify steps using relevant physical laws, theorems, or mathematical identities."
        replacement = "Justify steps using relevant physical laws, theorems, mathematical identities or numerical codes."
        system_prompt = system_prompt.replace(needle, replacement)

    if use_web_search:
        reminder = "\nYou must use web search engine to gather all the necessary information before solving the problem."
        system_prompt = f"{system_prompt}{reminder}{reminder}{reminder}"

    return system_prompt


class PromptBuilder:
    """
    Small helper that knows how to turn loaders/readers into the prompts needed by
    the solver.
    """

    def __init__(self, reader, parsing: bool, use_golden_for_prev_steps: bool, multiturn_with_answer: bool):
        self.reader = reader
        self.parsing = parsing
        self.use_golden_for_prev_steps = use_golden_for_prev_steps
        self.multiturn_with_answer = multiturn_with_answer

        # Cache expensive lookups so we can reuse them for every step.
        self._main_problem_list = reader.get_main_problem_list()
        self._sub_problem_list = reader.get_sub_problem_list()
        self._main_prompt = None
        self._sub_prompts: List[Optional[str]] = []

    def main_step(self) -> StepSpec:
        if self._main_prompt is None:
            self._main_prompt = self.reader.get_main_problem_prompt(
                self._main_problem_list,
                parsing=self.parsing,
            )
        return StepSpec(
            name="main",
            prompt=self._main_prompt,
            code_template=str(getattr(self.reader, "main_problem_code_template", "")),
        )

    def sub_step(self, index: int) -> StepSpec:
        while len(self._sub_prompts) <= index:
            self._sub_prompts.append(None)

        if self._sub_prompts[index] is None:
            if self.use_golden_for_prev_steps and not self.multiturn_with_answer:
                prompt = self.reader.get_sub_problem_prompt(
                    self._sub_problem_list,
                    from_step=0,
                    until_step=index + 1,
                    parsing=self.parsing,
                )
            else:
                prompt = self.reader.get_sub_problem_prompt(
                    self._sub_problem_list,
                    from_step=index,
                    until_step=index + 1,
                    parsing=self.parsing,
                    multiturn_with_answer=self.multiturn_with_answer,
                )
            self._sub_prompts[index] = prompt

        code_templates = list(getattr(self.reader, "sub_problem_code_templates", []))
        code_template = code_templates[index] if index < len(code_templates) else ""

        return StepSpec(
            name=f"sub_{index}",
            prompt=self._sub_prompts[index] or "",
            code_template=str(code_template),
        )
