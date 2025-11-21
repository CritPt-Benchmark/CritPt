from pathlib import Path
from critpt.data_loader import NotebookDataLoader

path = Path('../data/quantum_error_correction_main.ipynb')
print(f'Loading: {path}')
print(f'Exists: {path.exists()}')

if path.exists():
    loader = NotebookDataLoader(path)
    print(f'Dataset name: {loader.dataset_name}')
    
    # Try loading problems
    try:
        problems = loader.load_all_problems()
        print(f'Loaded {len(problems)} problems')
        for p in problems:
            print(f'  - Problem ID: {p.problem_id}')
            print(f'    Has answer: {p.golden_answer is not None}')
    except Exception as e:
        print(f'Error loading problems: {e}')
        import traceback
        traceback.print_exc()