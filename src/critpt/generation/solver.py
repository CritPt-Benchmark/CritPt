from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, TaskState, solver

from critpt.generation.artifacts import load_cached_completion, save_step_artifact
from critpt.generation.prompts import PromptBuilder, StepSpec, build_system_prompt
from critpt.generation.solve_with_parse import solve_with_parse

import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    spec: StepSpec
    processed_state: Optional[TaskState]
    parsing_state: Optional[TaskState]

    def completion(self) -> Optional[str]:
        output = getattr(self.parsing_state, "output", None)
        if output and getattr(output, "completion", None):
            return output.completion
        return None


async def _run_step_for_spec(
    generate: Generate,
    base_state: TaskState,
    spec: StepSpec,
    parsing: bool,
) -> tuple[StepResult, TaskState]:
    working_state = copy.deepcopy(base_state)
    working_state.messages.append(ChatMessageUser(content=spec.prompt))
    processed_state, parsing_state = await solve_with_parse(
        generate,
        working_state,
        parsing,
        spec.code_template,
    )
    return StepResult(spec, processed_state, parsing_state), working_state


def _determine_output_dir(state: TaskState, output_dir: Optional[Path]) -> Optional[Path]:
    _output_dir = output_dir or state.metadata.get("output_dir")
    if _output_dir is None:
        return None

    _output_dir = _output_dir / state.model.name
    if state.metadata.get("model_label"):
        _output_dir = _output_dir / state.metadata["model_label"]
    if state.metadata.get("tool_label"):
        _output_dir = _output_dir / state.metadata["tool_label"]

    epoch = state.epoch - 1
    return _output_dir / str(epoch)


def _maybe_load_cached_results(
    state: TaskState,
    specs: List[StepSpec],
    output_dir: Optional[Path],
) -> Optional[List[str]]:
    if not state.metadata.get("skip_if_exists") or output_dir is None or not specs:
        return None

    cached_completions: List[str] = []
    for spec in specs:
        completion = load_cached_completion(output_dir, spec.name, spec.problem_id)
        if completion is None:
            return None
        cached_completions.append(completion)

    return cached_completions


@solver
def cortex_solver(
    use_golden_for_prev_steps: bool = None,
    output_dir: Optional[Path] = None,
    parsing: bool = None,
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _parsing = parsing if parsing is not None else state.metadata["config"]["parsing"]
        _use_golden_for_prev_steps = (
            use_golden_for_prev_steps
            if use_golden_for_prev_steps is not None
            else state.metadata["config"]["use_golden_for_prev_steps"]
        )
        _multiturn_with_answer = state.metadata.get("multiturn_with_answer")

        state.messages.clear()
        original_state = copy.deepcopy(state)

        reader = state.metadata["reader"]
        problem_ids = state.metadata.get("problem_ids", [])
        n_sub_problems = len(reader.sub_problems)
        if state.metadata.get("num_steps") is not None:
            n_sub_problems = state.metadata["num_steps"]

        run_main = state.metadata.get("run_main", True)
        run_sub = state.metadata.get("run_sub", True)

        prompt_builder = PromptBuilder(
            reader,
            _parsing,
            _use_golden_for_prev_steps,
            _multiturn_with_answer,
        )

        specs: List[StepSpec] = []
        main_spec: Optional[StepSpec] = None
        if run_main:
            main_spec = prompt_builder.main_step()
            if problem_ids:
                main_spec.problem_id = problem_ids[0]
            specs.append(main_spec)

        sub_specs: List[StepSpec] = []
        if run_sub:
            for idx in range(n_sub_problems):
                spec = prompt_builder.sub_step(idx)
                if idx + 1 < len(problem_ids):
                    spec.problem_id = problem_ids[idx + 1]
                sub_specs.append(spec)
            specs.extend(sub_specs)

        generation_output_dir = _determine_output_dir(state, output_dir)
        cached = _maybe_load_cached_results(state, specs, generation_output_dir)
        if cached:
            logger.info("SKIP: cached completions found")
            for completion in cached:
                original_state.messages.append(ChatMessageAssistant(content=completion))
            return original_state

        if state.metadata.get("skip_if_not_exists"):
            logger.info("SKIP: missing cached result but skip_if_not_exists=True")
            return original_state

        system_prompt = build_system_prompt(
            _parsing,
            state.metadata.get("use_python"),
            state.metadata.get("use_web_search"),
        )
        logger.debug("System prompt prepared")
        state.messages.append(ChatMessageSystem(content=system_prompt))

        main_result: Optional[StepResult] = None
        if run_main and main_spec:
            main_result, _ = await _run_step_for_spec(generate, state, main_spec, _parsing)

        sub_results: List[StepResult] = []
        if run_sub and sub_specs:
            sequential = (not _use_golden_for_prev_steps) or _multiturn_with_answer
            if sequential:
                for spec in sub_specs:
                    result, state = await _run_step_for_spec(generate, state, spec, _parsing)
                    sub_results.append(result)
            else:
                async def _detached(spec: StepSpec) -> StepResult:
                    result, _ = await _run_step_for_spec(generate, state, spec, _parsing)
                    return result

                sub_results = list(
                    await asyncio.gather(*[_detached(spec) for spec in sub_specs])
                )

        if generation_output_dir:
            if main_result:
                save_step_artifact(
                    generation_output_dir,
                    main_result.spec.name,
                    main_result.processed_state,
                    main_result.parsing_state,
                    main_result.spec.prompt,
                )
            for result in sub_results:
                save_step_artifact(
                    generation_output_dir,
                    result.spec.name,
                    result.processed_state,
                    result.parsing_state,
                    result.spec.prompt,
                )

        if main_result and main_result.completion():
            original_state.messages.append(
                ChatMessageAssistant(content=main_result.completion())
            )

        for result in sub_results:
            completion = result.completion()
            if completion:
                original_state.messages.append(ChatMessageAssistant(content=completion))

        logger.debug("SOLVER COMPLETE")
        return original_state

    return solve
