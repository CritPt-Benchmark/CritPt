"""
Notebook-based data loader.
Wraps the existing Reader class to conform to BaseDataLoader interface.
"""
from pathlib import Path
from typing import List, Optional

from critpt.notebook_reader import Reader
from .base_loader import BaseDataLoader, ProblemData


class NotebookDataLoader(BaseDataLoader):
    """Data loader for Jupyter notebook format."""

    def __init__(self, notebook_path: Path):
        super().__init__(notebook_path)
        self.reader = Reader(notebook_path)

    def _extract_dataset_name(self) -> str:
        """Extract dataset name from notebook filename."""
        # Remove file extension and special characters
        name = self.source_path.stem
        # Convert to clean identifier (remove spaces, special chars)
        clean_name = name.replace(" ", "_").replace("-", "_")
        # Remove consecutive underscores
        while "__" in clean_name:
            clean_name = clean_name.replace("__", "_")
        return clean_name

    def load_main_problem(self) -> ProblemData:
        """Load main problem from notebook."""
        problem_id = self.generate_problem_id("main")

        # Get problem description from reader
        problem_description = self.reader.get_main_problem_prompt(
            self.reader.get_main_problem_list(),
            parsing=False
        )

        # Get settings/metadata
        settings = self.reader.get_settings('main')

        return ProblemData(
            problem_id=problem_id,
            problem_type="main",
            problem_index=None,
            problem_description=problem_description,
            code_template=str(self.reader.main_problem_code_template) if self.reader.main_problem_code_template else "",
            answer_code=self.reader.main_problem_answer_code or "",
            answer_only_code=str(self.reader.main_problem_answer_only_code) if self.reader.main_problem_answer_only_code else "",
            testcases=self.reader.main_testcases,
            metadata={
                "settings": settings,
                "tag": str(self.reader.tag) if self.reader.tag else "",
                "problem_setup": str(self.reader.problem_setup) if self.reader.problem_setup else "",
                "notebook_path": str(self.source_path),
            }
        )

    def load_sub_problems(self) -> List[ProblemData]:
        """Load all sub-problems from notebook."""
        sub_problems = []

        for i in range(len(self.reader.sub_problems)):
            problem_id = self.generate_problem_id("sub", i)

            # Get problem description
            problem_description = self.reader.get_sub_problem_prompt(
                self.reader.get_sub_problem_list(),
                until_step=i,
                parsing=False
            )

            # Get settings/metadata
            settings = self.reader.get_settings(i)

            sub_problem = ProblemData(
                problem_id=problem_id,
                problem_type="sub",
                problem_index=i,
                problem_description=problem_description,
                code_template=str(self.reader.sub_problem_code_templates[i]) if self.reader.sub_problem_code_templates[i] else "",
                answer_code=self.reader.sub_problem_answer_codes[i] or "",
                answer_only_code=str(self.reader.sub_problem_answer_only_codes[i]) if self.reader.sub_problem_answer_only_codes[i] else "",
                testcases=self.reader.sub_testcases[i],
                metadata={
                    "settings": settings,
                    "sub_problem_index": i,
                    "notebook_path": str(self.source_path),
                }
            )
            sub_problems.append(sub_problem)

        return sub_problems

    def get_problem_by_id(self, problem_id: str) -> Optional[ProblemData]:
        """Retrieve a specific problem by its ID."""
        # Parse problem_id to determine type
        parts = problem_id.split("_")

        # Check if it matches our dataset
        if not problem_id.startswith(self.dataset_name):
            return None

        # Extract type from end of ID
        if problem_id.endswith("_main"):
            return self.load_main_problem()
        elif "_sub_" in problem_id:
            # Extract sub-problem index
            try:
                sub_idx = int(parts[-1])
                sub_problems = self.load_sub_problems()
                if 0 <= sub_idx < len(sub_problems):
                    return sub_problems[sub_idx]
            except (ValueError, IndexError):
                return None

        return None

    # Expose the underlying Reader for backward compatibility
    def get_reader(self) -> Reader:
        """Get the underlying Reader object for backward compatibility."""
        return self.reader
