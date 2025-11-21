"""
Convert all notebooks in data directory to JSON format.

This script processes all .ipynb files in the data directory and converts them
to a standardized JSON format that can be loaded by the JsonDataLoader.
"""
import json
from pathlib import Path
from critpt.data_loader.notebook_loader import NotebookDataLoader


def convert_notebook_to_json(notebook_path: Path, output_dir: Path):
    """
    Convert a single notebook to JSON format.

    Args:
        notebook_path: Path to the notebook file
        output_dir: Directory to save the JSON output
    """
    print(f"Processing: {notebook_path.name}")

    try:
        # Load notebook using NotebookDataLoader
        loader = NotebookDataLoader(notebook_path)

        # Extract all problems
        all_problems = loader.load_all_problems()

        # Convert to JSON-serializable format
        json_data = {
            "dataset_name": loader.dataset_name,
            "source_notebook": str(notebook_path),
            "problems": [problem.to_private_dict() for problem in all_problems]
        }

        # Save to JSON file
        output_file = output_dir / f"{loader.dataset_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved to: {output_file.name}")
        print(f"  ✓ Problems: {len(all_problems)} (1 main + {len(all_problems)-1} sub-problems)")

    except Exception as e:
        print(f"  ✗ Error processing {notebook_path.name}: {e}")
        import traceback
        traceback.print_exc()


def convert_all_notebooks(data_dir: Path, output_dir: Path = None):
    """
    Convert all notebooks in the data directory to JSON.

    Args:
        data_dir: Directory containing notebook files
        output_dir: Directory to save JSON files (defaults to data_dir/json)
    """
    if output_dir is None:
        output_dir = data_dir / "json"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all notebook files
    notebook_files = list(data_dir.glob("*.ipynb"))

    if not notebook_files:
        print(f"No notebook files found in {data_dir}")
        return

    print(f"Found {len(notebook_files)} notebook(s) to convert")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Convert each notebook
    for notebook_path in notebook_files:
        convert_notebook_to_json(notebook_path, output_dir)
        print()

    print("-" * 60)
    print(f"Conversion complete! JSON files saved to: {output_dir}")


if __name__ == "__main__":
    # Default paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "example_challenges"
    print(data_dir)

    # Convert all notebooks
    convert_all_notebooks(data_dir)
