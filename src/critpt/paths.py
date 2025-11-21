import os
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
TEST_DIR = PROJECT_ROOT / 'test_dir'
SRC_ROOT = PROJECT_ROOT / "src"
TEMPLATES_ROOT = PROJECT_ROOT / "templates"
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"

if __name__ == '__main__':

    print(PROJECT_ROOT)
    print(DATA_DIR)
    print(RESULTS_DIR)
    print(TEST_DIR)