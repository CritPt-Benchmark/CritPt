"""
Generation-only mode (PUBLIC VERSION).

This module provides generation functionality without evaluation.
It outputs submission files with problem IDs for later evaluation.
"""
import asyncio
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem
from inspect_ai.scorer import Score, Target, scorer, mean
from inspect_ai.solver import TaskState, use_tools
from inspect_ai.tool import bash, python, web_search

from critpt.data_loader import NotebookDataLoader, JsonDataLoader
from critpt.generation.solver import cortex_solver
from critpt.paths import RESULTS_DIR, PROJECT_ROOT
from critpt.submission import create_submission
from critpt.templates import get_default_system_prompt

try:
    from critpt.evaluation.eval_client import evaluate_submission_async
    EVAL_CLIENT_AVAILABLE = True
except ImportError:
    EVAL_CLIENT_AVAILABLE = False
    print("Warning: Evaluation client not available. Online evaluation will be disabled.")

CMD_TIMEOUT = 1800


@scorer(metrics=[{"Generation Complete": [mean()]}])
def generation_only_scorer(output_dir=None, evaluate_online=False, server_url=None, **params):
    """
    Scorer for generation-only mode.
    Saves submissions instead of evaluating them.

    Args:
        output_dir: Directory to save submissions
        evaluate_online: If True, send submissions to evaluation server
        server_url: URL of evaluation server (required if evaluate_online=True)
    """
    async def score(state: TaskState, target: Target):
        _output_dir = output_dir or state.metadata.get('output_dir')
        if _output_dir is not None:
            _output_dir = _output_dir / state.model.name
            if state.metadata.get("model_label"):
                _output_dir = _output_dir / state.metadata["model_label"]
            # Add tool_label (code, code_and_search, etc.) to path
            if state.metadata.get("tool_label"):
                _output_dir = _output_dir / state.metadata["tool_label"]

        epoch = state.epoch - 1
        print(f"GENERATION SCORER {_output_dir}: Epoch {epoch}")
        if _output_dir is not None:
            _output_dir = _output_dir / str(epoch)
            _output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare evaluation output directory (mirrors generation structure but under evaluations/)
        _eval_output_dir = None
        if evaluate_online and _output_dir is not None:
            # Convert generations path to evaluations path
            _eval_output_dir = Path(str(_output_dir).replace('/generations/', '/evaluations/'))
            _eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Get problem metadata
        loader = state.metadata["loader"]
        problem_ids = state.metadata["problem_ids"]
        results = state.messages

        if len(results) == 0 and state.metadata.get('skip_if_not_exists'):
            return Score(value={"Generation Complete": 0})

        # Check if submissions already exist when skip_if_exists is enabled
        if state.metadata.get('skip_if_exists') and _output_dir:
            all_exist = True

            # Get run_main and run_sub flags to only check files we plan to generate
            run_main = state.metadata.get("run_main", True)
            run_sub = state.metadata.get("run_sub", True)

            # Check main problem only if we plan to run it
            if run_main and len(problem_ids) > 0:
                main_submission_file = _output_dir / f"{problem_ids[0]}.json"
                if not main_submission_file.exists():
                    all_exist = False

            # Check sub-problems only if we plan to run them
            if all_exist and run_sub:
                n_sub_problems = len(problem_ids) - 1
                if state.metadata.get("num_steps") is not None:
                    n_sub_problems = min(n_sub_problems, state.metadata["num_steps"])

                for sub_idx in range(n_sub_problems):
                    if sub_idx + 1 < len(problem_ids):
                        sub_submission_file = _output_dir / f"{problem_ids[sub_idx + 1]}.json"
                        if not sub_submission_file.exists():
                            all_exist = False
                            break

            # If all submission files already exist, skip this sample
            # Only skip if we're actually planning to generate something (run_main or run_sub)
            if all_exist and (run_main or run_sub) and len(problem_ids) > 0:
                print(f"SKIP: All submission files already exist in {_output_dir}")
                return Score(value={"Generation Complete": 1.0})

        # Get generation config
        generation_config = {
            "use_golden_for_prev_steps": state.metadata['config']['use_golden_for_prev_steps'],
            "parsing": state.metadata['config']['parsing'],
            "multiturn_with_answer": state.metadata.get('multiturn_with_answer'),
            "use_python": state.metadata.get('use_python'),
            "use_web_search": state.metadata.get('use_web_search'),
        }

        model_name = f"{state.model.api}/{state.model.name}"

        # Get run_main and run_sub flags
        run_main = state.metadata.get("run_main", True)
        run_sub = state.metadata.get("run_sub", True)

        # Save submissions for each problem
        submissions_saved = 0
        result_index = 0  # Track which result we're on

        # Main problem
        if run_main and len(results) > result_index and len(problem_ids) > 0:
            main_result = results[result_index]
            main_problem_id = problem_ids[0]
            result_index += 1

            # Convert messages to serializable format
            messages_data = []
            for msg in state.messages:
                msg_dict = {
                    "role": msg.role,
                    "content": str(msg.content),
                }
                messages_data.append(msg_dict)

            # Create submission
            submission = create_submission(
                problem_id=main_problem_id,
                generated_code=str(main_result.content),
                model=model_name,
                generation_config=generation_config,
                messages=messages_data
            )

            # Save to file
            submission_file = _output_dir / f"{main_problem_id}.json"
            submission.to_json(submission_file)
            print(f"Saved submission: {submission_file}")
            submissions_saved += 1

            # Online evaluation if enabled
            if evaluate_online and EVAL_CLIENT_AVAILABLE and server_url:
                eval_file = _eval_output_dir / f"{main_problem_id}_eval.json"

                # Skip if evaluation result already exists (based on skip_if_exists setting)
                if state.metadata.get('skip_if_exists') and eval_file.exists():
                    print(f"SKIP: Evaluation result already exists for {main_problem_id}: {eval_file}")
                else:
                    try:
                        print(f"Evaluating {main_problem_id} online...")
                        eval_result = await evaluate_submission_async(server_url, submission)
                        print(f"Evaluation result for {main_problem_id}:")
                        print(f"  Score: {eval_result['score']}")
                        print(f"  Judge Result: {eval_result['judge_result']}")

                        # Save evaluation result to evaluations directory (not alongside submission)
                        with open(eval_file, 'w', encoding='utf-8') as f:
                            json.dump(eval_result, f, indent=2, ensure_ascii=False)
                        print(f"Saved evaluation result: {eval_file}")
                    except Exception as e:
                        print(f"Warning: Online evaluation failed for {main_problem_id}: {e}")
            elif evaluate_online and not EVAL_CLIENT_AVAILABLE:
                print(f"Warning: Online evaluation requested but eval_client not available")
            elif evaluate_online and not server_url:
                print(f"Warning: Online evaluation requested but server_url not provided")

        # Sub-problems
        if run_sub:
            n_sub_problems = len(problem_ids) - 1
            if state.metadata.get("num_steps") is not None:
                n_sub_problems = min(n_sub_problems, state.metadata["num_steps"])

            for sub_step in range(n_sub_problems):
                if result_index < len(results) and sub_step + 1 < len(problem_ids):
                    sub_result = results[result_index]
                    sub_problem_id = problem_ids[sub_step + 1]
                    result_index += 1

                    submission = create_submission(
                        problem_id=sub_problem_id,
                        generated_code=str(sub_result.content),
                        model=model_name,
                        generation_config=generation_config,
                        messages=None  # Don't save full messages for sub-problems to save space
                    )

                    submission_file = _output_dir / f"{sub_problem_id}.json"
                    submission.to_json(submission_file)
                    print(f"Saved submission: {submission_file}")
                    submissions_saved += 1

                    # Online evaluation if enabled
                    if evaluate_online and EVAL_CLIENT_AVAILABLE and server_url:
                        eval_file = _eval_output_dir / f"{sub_problem_id}_eval.json"

                        # Skip if evaluation result already exists (based on skip_if_exists setting)
                        if state.metadata.get('skip_if_exists') and eval_file.exists():
                            print(f"SKIP: Evaluation result already exists for {sub_problem_id}: {eval_file}")
                        else:
                            try:
                                print(f"Evaluating {sub_problem_id} online...")
                                eval_result = await evaluate_submission_async(server_url, submission)
                                print(f"Evaluation result for {sub_problem_id}:")
                                print(f"  Score: {eval_result['score']}")
                                print(f"  Judge Result: {eval_result['judge_result']}")

                                # Save evaluation result to evaluations directory (not alongside submission)
                                with open(eval_file, 'w', encoding='utf-8') as f:
                                    json.dump(eval_result, f, indent=2, ensure_ascii=False)
                                print(f"Saved evaluation result: {eval_file}")
                            except Exception as e:
                                print(f"Warning: Online evaluation failed for {sub_problem_id}: {e}")

        # Save batch metadata
        batch_metadata = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "generation_config": generation_config,
            "num_submissions": submissions_saved,
            "problem_ids": problem_ids,
        }
        metadata_file = _output_dir / "batch_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, indent=2, ensure_ascii=False)

        return Score(value={"Generation Complete": 1.0 if submissions_saved > 0 else 0.0})

    return score


def critpt_generate(
    reader_paths,
    output_dir_prefix=None,
    use_golden_for_prev_steps=None,
    parsing=None,
    multiturn_with_answer=None,
    skip_if_exists=True,
    skip_if_not_exists=False,
    model_label="",
    debug_num_steps=None,
    use_python=False,
    use_web_search=False,
    debug_num_samples=None,
    evaluate=False,
    server_url=None,
    run_main=True,
    run_sub=True
):
    """
    Create a generation-only task (PUBLIC VERSION).

    This creates a Task that generates solutions and saves them as submissions,
    without performing evaluation.

    Args:
        reader_paths: Paths to problem data (notebooks or JSON)
        output_dir_prefix: Directory to save submissions
        use_golden_for_prev_steps: Use golden answers for previous steps
        parsing: Enable parsing mode
        multiturn_with_answer: Include answers in multi-turn conversation
        skip_if_exists: Skip if results already exist
        skip_if_not_exists: Skip if results don't exist
        model_label: Label for model variant
        debug_num_steps: Limit number of sub-problems for debugging
        use_python: Enable Python tool
        use_web_search: Enable web search tool
        debug_num_samples: Limit number of samples for debugging
        evaluate: If True, send submissions to evaluation server after generation
        server_url: URL of evaluation server (e.g., "http://localhost:8000")
        run_main: If True, run main problem (default: True)
        run_sub: If True, run sub-problems (default: True)

    Returns:
        inspect_ai.Task for generation
    """
    if output_dir_prefix is None:
        output_dir_prefix = RESULTS_DIR / "generations"

    print("CORTEX GENERATE: SAVE DIR PREFIX", output_dir_prefix)

    if debug_num_samples is not None:
        reader_paths = reader_paths[:debug_num_samples]

    # Load data using new data loader abstraction
    loaders = []
    for path in reader_paths:
        full_path = (PROJECT_ROOT / Path(path)).absolute()
        print("FULL PATH", full_path)
        path_list = glob.glob(str(full_path))
        for _path in path_list:
            _path = Path(_path)
            file_path = Path(_path)
            print("File path:", file_path)

            # Support both JSON and notebook formats, defaulting to JSON
            if _path.suffix == '.json':
                loader = JsonDataLoader(file_path)
                loaders.append(loader)
            elif _path.suffix == '.ipynb':
                loader = NotebookDataLoader(file_path)
                loaders.append(loader)
            else:
                print(f"Warning: Unsupported file format for {file_path}. Only .json and .ipynb files are supported.")

    def create_sample(
        _use_golden_for_prev_steps,
        _parsing,
        multiturn_with_answer,
        _loader,
        _skip_if_exists,
        _skip_if_not_exists,
        use_python,
        use_web_search,
        run_main,
        run_sub
    ):
        # Get reader for backward compatibility
        _reader = _loader.get_reader()

        # Generate problem IDs
        main_problem = _loader.load_main_problem()
        sub_problems = _loader.load_sub_problems()

        problem_ids = [main_problem.problem_id]
        problem_ids.extend([sp.problem_id for sp in sub_problems])

        # Determine save dir
        config = dict(
            use_golden_for_prev_steps=_use_golden_for_prev_steps,
            parsing=_parsing,
            multiturn_with_answer=multiturn_with_answer
        )

        output_dir = output_dir_prefix / _loader.dataset_name
        for k, v in config.items():
            output_dir = output_dir / (str(k) + "_" + str(v))

        # Create tool label based on use_python and use_web_search
        tool_label = None
        if use_web_search and use_python:
            tool_label = "code_and_search"
        elif use_python:
            tool_label = "code"
        elif use_web_search:
            # This shouldn't happen according to user, but handle it anyway
            tool_label = "search"

        config_meta = {
            "reader": _reader,  # Keep for backward compat
            "loader": _loader,
            "problem_ids": problem_ids,
            "skip_if_exists": _skip_if_exists,
            "skip_if_not_exists": _skip_if_not_exists,
            "config": config,
            'output_dir': output_dir,
            "model_label": model_label,
            "tool_label": tool_label,  # Add tool_label to metadata
            "multiturn_with_answer": multiturn_with_answer,
            "use_python": use_python,
            "use_web_search": use_web_search,
            "run_main": run_main,
            "run_sub": run_sub,
        }

        if debug_num_steps is not None:
            config_meta["num_steps"] = debug_num_steps

        if debug_num_samples is not None:
            config_meta["num_samples"] = debug_num_steps

        # Create sample (using Reader for backward compat with existing solver)
        sample = Sample(
            input=str(_reader.dummy),
            metadata=config_meta,
            target=[str(_reader.main_problem_answer_only_code)] +
                   [str(sub_problem_answer_code) for sub_problem_answer_code in _reader.sub_problem_answer_codes],
        )
        return sample

    # Build dataset
    dataset = []

    for loader in loaders:
        if use_golden_for_prev_steps is not None:
            _use_golden_for_prev_steps = [use_golden_for_prev_steps]
        else:
            _use_golden_for_prev_steps = [False, True]

        if parsing is not None:
            _parsing = [parsing]
        else:
            _parsing = [False, True]

        for __use_golden_for_prev_steps in _use_golden_for_prev_steps:
            for __parsing in _parsing:
                sample = create_sample(
                    __use_golden_for_prev_steps,
                    __parsing,
                    multiturn_with_answer,
                    loader,
                    skip_if_exists,
                    skip_if_not_exists,
                    use_python,
                    use_web_search,
                    run_main,
                    run_sub
                )
                dataset.append(sample)

    # Build solver list
    solver_list = []

    if use_python or use_web_search:
        tool_list = []

        if use_python:
            tool_list.extend([
                bash(CMD_TIMEOUT),
                python(CMD_TIMEOUT)
            ])
        if use_web_search:
            tool_list.append(
                web_search("openai")
            )

        solver_list.append(use_tools(tool_list))

    solver_list.append(cortex_solver())

    return Task(
        dataset=dataset,
        solver=solver_list,
        scorer=generation_only_scorer(evaluate_online=evaluate, server_url=server_url),
        sandbox="local",
    )
