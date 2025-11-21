"""
JSON-based data loader with robust handling of missing keys.
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from .base_loader import BaseDataLoader, ProblemData


class JsonDataLoader(BaseDataLoader):
    """
    Data loader for JSON format with robust handling of missing keys.

    This loader expects JSON files with the following structure:
    {
        "dataset_name": "...",
        "source_notebook": "...",
        "problems": [
            {
                "problem_id": "...",
                "problem_type": "main" or "sub",
                "problem_index": null or int,
                "problem_description": "...",
                "code_template": "...",
                "answer_code": "...",
                "answer_only_code": "...",
                "testcases": [...],
                "metadata": {...}
            },
            ...
        ]
    }
    """

    def __init__(self, json_path: Path):
        """
        Initialize JSON data loader.

        Args:
            json_path: Path to the JSON file
        """
        # Initialize data before calling super().__init__()
        # because parent's __init__ calls _extract_dataset_name()
        self._data = None
        self._problems_cache = None
        super().__init__(json_path)
        self._load_json()

    def _load_json(self):
        """Load and parse JSON file."""
        try:
            with open(self.source_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {self.source_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {self.source_path}: {e}")

    def _extract_dataset_name(self) -> str:
        """Extract dataset name from JSON or filename."""
        if self._data and "dataset_name" in self._data:
            return self._data["dataset_name"]
        # Fallback to filename
        return self.source_path.stem

    def _parse_problem(self, problem_dict: Dict[str, Any]) -> ProblemData:
        """
        Parse a problem dictionary into ProblemData with robust handling of missing keys.

        Args:
            problem_dict: Dictionary containing problem data

        Returns:
            ProblemData object with default values for missing keys
        """
        # Helper function to safely get values with defaults
        def get_str(key: str, default: str = "") -> str:
            value = problem_dict.get(key, default)
            return str(value) if value is not None else default

        def get_int(key: str, default: Optional[int] = None) -> Optional[int]:
            value = problem_dict.get(key, default)
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        def get_list(key: str, default: Optional[List] = None) -> Optional[List]:
            value = problem_dict.get(key, default)
            if value is None:
                return default
            if isinstance(value, list):
                return value
            return default

        def get_dict(key: str, default: Optional[Dict] = None) -> Dict:
            value = problem_dict.get(key, default)
            if value is None:
                return default if default is not None else {}
            if isinstance(value, dict):
                return value
            return default if default is not None else {}

        # Extract problem type and index
        problem_type = get_str("problem_type", "main")
        problem_index = get_int("problem_index", None)

        # Generate problem_id if missing
        problem_id = get_str("problem_id")
        if not problem_id:
            problem_id = self.generate_problem_id(problem_type, problem_index)

        return ProblemData(
            problem_id=problem_id,
            problem_type=problem_type,
            problem_index=problem_index,
            problem_description=get_str("problem_description"),
            code_template=get_str("code_template"),
            answer_code=get_str("answer_code"),
            answer_only_code=get_str("answer_only_code"),
            testcases=get_list("testcases"),
            metadata=get_dict("metadata")
        )

    def _get_all_problems(self) -> List[ProblemData]:
        """Load and cache all problems from JSON."""
        if self._problems_cache is not None:
            return self._problems_cache

        if not self._data:
            raise ValueError("JSON data not loaded")

        problems_list = self._data.get("problems", [])
        if not isinstance(problems_list, list):
            raise ValueError("'problems' field in JSON must be a list")

        self._problems_cache = [
            self._parse_problem(problem_dict)
            for problem_dict in problems_list
        ]

        return self._problems_cache

    def load_main_problem(self) -> ProblemData:
        """
        Load main problem from JSON.

        Returns:
            ProblemData for the main problem

        Raises:
            ValueError: If no main problem is found
        """
        all_problems = self._get_all_problems()

        for problem in all_problems:
            if problem.problem_type == "main":
                return problem

        raise ValueError(f"No main problem found in {self.source_path}")

    def load_sub_problems(self) -> List[ProblemData]:
        """
        Load all sub-problems from JSON.

        Returns:
            List of ProblemData for sub-problems (may be empty)
        """
        all_problems = self._get_all_problems()

        sub_problems = [
            problem for problem in all_problems
            if problem.problem_type == "sub"
        ]

        # Sort by problem_index if available
        sub_problems.sort(key=lambda p: p.problem_index if p.problem_index is not None else 0)

        return sub_problems

    def get_problem_by_id(self, problem_id: str) -> Optional[ProblemData]:
        """
        Retrieve a specific problem by its ID.

        Args:
            problem_id: The problem identifier

        Returns:
            ProblemData if found, None otherwise
        """
        all_problems = self._get_all_problems()

        for problem in all_problems:
            if problem.problem_id == problem_id:
                return problem

        return None

    def reload(self):
        """Reload the JSON file and clear cache."""
        self._problems_cache = None
        self._load_json()

    # ============================================================================
    # Backward compatibility: Reader interface for existing code
    # Note: JSON data is already preprocessed, so we ignore notebook-specific
    # parameters like parsing, skip_unparsed, multiturn_with_answer
    # ============================================================================

    def get_reader(self):
        """Return self for backward compatibility with code expecting Reader."""
        return self

    @property
    def dummy(self):
        """Return empty string (Reader interface - used as initial input in Sample)."""
        return ""

    # Main problem methods
    def get_main_problem_list(self) -> List[str]:
        """Return list containing main problem description."""
        try:
            main = self.load_main_problem()
            return [main.problem_description] if main and main.problem_description else []
        except ValueError:
            return []

    def get_main_problem_prompt(self, problem_list: List = None, parsing: bool = False) -> str:
        """Return main problem description. Params are for Reader compatibility only."""
        try:
            main = self.load_main_problem()
            return main.problem_description or ""
        except ValueError:
            return ""

    @property
    def main_problem_code_template(self) -> str:
        """Return main problem code template."""
        try:
            main = self.load_main_problem()
            return main.code_template or ""
        except ValueError:
            return ""

    @property
    def main_problem_answer_only_code(self) -> str:
        """Return main problem answer-only code."""
        try:
            main = self.load_main_problem()
            return main.answer_only_code or ""
        except ValueError:
            return ""

    # Sub-problem methods
    @property
    def sub_problems(self) -> List[ProblemData]:
        """Return all sub-problems."""
        return self.load_sub_problems()

    def get_sub_problem_list(self) -> List[str]:
        """Return list of sub-problem descriptions."""
        return [sp.problem_description for sp in self.load_sub_problems()
                if sp.problem_description]

    def get_sub_problem_prompt(self, problem_list: List = None, from_step: int = 0,
                               until_step: Optional[int] = None, parsing: bool = False,
                               skip_unparsed: bool = True, multiturn_with_answer: bool = False) -> str:
        """
        Return combined sub-problem descriptions.

        Note: parsing/skip_unparsed/multiturn_with_answer are notebook-specific
        and ignored for JSON data which is already preprocessed.
        """
        sub_probs = self.load_sub_problems()
        if not sub_probs:
            return ""

        # Determine range
        end_idx = until_step if until_step is not None else len(sub_probs)
        end_idx = min(end_idx, len(sub_probs))

        # Combine descriptions in range
        descriptions = [sub_probs[i].problem_description
                       for i in range(from_step, end_idx)
                       if i < len(sub_probs) and sub_probs[i].problem_description]

        return "\n\n".join(descriptions)

    @property
    def sub_problem_answer_codes(self) -> List[str]:
        """Return list of sub-problem answer codes."""
        return [sp.answer_code or "" for sp in self.load_sub_problems()]

    @property
    def sub_problem_answer_only_codes(self) -> List[str]:
        """Return list of sub-problem answer-only codes."""
        return [sp.answer_only_code or "" for sp in self.load_sub_problems()]

    @property
    def sub_problem_code_templates(self) -> List[str]:
        """Return list of sub-problem code templates."""
        return [sp.code_template or "" for sp in self.load_sub_problems()]
