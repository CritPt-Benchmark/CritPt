"""
Abstract base class for data loaders.
This allows easy switching between notebook format and JSON/HuggingFace datasets.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ProblemData:
    """Standardized problem data structure."""

    # Problem identification
    problem_id: str  # Format: {dataset_name}_{type}_{index}, e.g., "EC_quantum_main" or "EC_quantum_sub_0"
    problem_type: str  # "main" or "sub"
    problem_index: Optional[int]  # None for main, 0,1,2... for sub-problems

    # Problem content (public)
    problem_description: str  # The problem text
    code_template: str  # Template code for student to fill

    # Golden answers (private - only for evaluation)
    answer_code: str  # Complete answer code
    answer_only_code: str  # Answer code without template
    testcases: Optional[List[Dict[str, Any]]]  # Test cases for validation

    # Additional metadata
    metadata: Dict[str, Any]  # Extra settings, tags, etc.

    def to_public_dict(self) -> Dict[str, Any]:
        """Return public version without golden answers."""
        return {
            "problem_id": self.problem_id,
            "problem_type": self.problem_type,
            "problem_index": self.problem_index,
            "problem_description": self.problem_description,
            "code_template": self.code_template,
            "metadata": self.metadata,
        }

    def to_private_dict(self) -> Dict[str, Any]:
        """Return full version with golden answers."""
        return {
            "problem_id": self.problem_id,
            "problem_type": self.problem_type,
            "problem_index": self.problem_index,
            "problem_description": self.problem_description,
            "code_template": self.code_template,
            "answer_code": self.answer_code,
            "answer_only_code": self.answer_only_code,
            "testcases": self.testcases,
            "metadata": self.metadata,
        }


class BaseDataLoader(ABC):
    """Abstract base class for loading problem data."""

    def __init__(self, source_path: Path):
        self.source_path = source_path
        self.dataset_name = self._extract_dataset_name()

    @abstractmethod
    def _extract_dataset_name(self) -> str:
        """Extract dataset name from source path."""
        pass

    @abstractmethod
    def load_main_problem(self) -> ProblemData:
        """Load main problem."""
        pass

    @abstractmethod
    def load_sub_problems(self) -> List[ProblemData]:
        """Load all sub-problems."""
        pass

    @abstractmethod
    def get_problem_by_id(self, problem_id: str) -> Optional[ProblemData]:
        """Retrieve a specific problem by its ID."""
        pass

    def load_all_problems(self) -> List[ProblemData]:
        """Load all problems (main + sub-problems)."""
        problems = [self.load_main_problem()]
        problems.extend(self.load_sub_problems())
        return problems

    def generate_problem_id(self, problem_type: str, index: Optional[int] = None) -> str:
        """
        Generate unique problem identifier.

        Format: {dataset_name}_{type}_{index}
        Examples:
            - EC_quantum_main
            - EC_quantum_sub_0
            - EC_quantum_sub_1
        """
        if problem_type == "main":
            return f"{self.dataset_name}_main"
        else:
            return f"{self.dataset_name}_sub_{index}"
