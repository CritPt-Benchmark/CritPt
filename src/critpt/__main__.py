"""
CRITPT Benchmark - Private CLI

Full CLI with generation AND evaluation.
This is a SUPERSET of the public package.
"""
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
ENV_FILE = PROJECT_ROOT / '.env'


def main():
    """Main entry point - supports both generation and evaluation."""

    # Check what mode user wants
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        # Evaluation modes
        if mode in ['eval', 'eval-batch', 'serve']:
            from critpt.evaluation.cli import main as eval_main
            eval_main()
            return

        # Generation mode
        elif mode == 'generate':
            # Remove 'generate' from args and run generation
            sys.argv.pop(1)

            from inspect_ai import eval as inspect_eval
            from critpt.critpt_generate import critpt_generate

            # Parse args
            kwargs = {}
            for arg in sys.argv[1:]:
                if "=" in arg:
                    k, v = arg.split("=", 1)
                    if k in ["epochs", "timeout", "max_tokens", "debug_num_steps", "debug_num_samples"]:
                        v = int(v)
                    if k in ["use_golden_for_prev_steps", "parsing", "skip_if_exists", "use_python", "use_web_search", "multiturn_with_answer", "evaluate", "run_main", "run_sub"]:
                        assert v in ["True", "False", "None"]
                        v = eval(v)
                    kwargs[k] = v

            load_dotenv(dotenv_path=ENV_FILE)

            # Load config
            task_config: dict | str | None = kwargs.pop('task_config', None)
            if not task_config or isinstance(task_config, str):
                if not task_config:
                    task_config_path = os.environ.get("CRITPT_TASK_CONFIG")
                else:
                    task_config_path = Path(task_config)
                if task_config_path and not Path(task_config_path).is_absolute():
                    task_config_path = PROJECT_ROOT / Path(task_config_path)
                if task_config_path:
                    with open(task_config_path, 'r', encoding='utf-8') as f:
                        task_config: dict = json.load(f)

            if not task_config:
                print("Error: No task config provided")
                print("Usage: python -m critpt generate task_config=path/to/config.json")
                return

            # List of parameters that belong to critpt_generate()
            critpt_generate_params = [
                'reader_paths', 'output_dir_prefix', 'use_golden_for_prev_steps',
                'parsing', 'multiturn_with_answer', 'skip_if_exists', 'skip_if_not_exists',
                'model_label', 'debug_num_steps', 'use_python', 'use_web_search',
                'debug_num_samples', 'evaluate', 'server_url', 'run_main', 'run_sub'
            ]

            # Move critpt_generate parameters from kwargs to task_config
            for k, v in kwargs.items():
                if k in critpt_generate_params:
                    task_config[k] = v

            # Remove critpt_generate parameters from kwargs
            for k in list(kwargs.keys()):
                if k in critpt_generate_params:
                    kwargs.pop(k)

            if task_config.get("multiturn_with_answer"):
                task_config["use_golden_for_prev_steps"] = False

            print("=" * 70)
            print("CRITPT BENCHMARK - GENERATION (PRIVATE)")
            print("=" * 70)
            task = critpt_generate(**task_config)
            inspect_eval(task, **kwargs)

            print("=" * 70)
            print("✓ Generation complete!")
            print("=" * 70)
            return

    # Show help
    print("=" * 70)
    print("CRITPT BENCHMARK - PRIVATE (FULL)")
    print("=" * 70)
    print()
    print("This package has BOTH generation and evaluation.")
    print()
    print("Usage:")
    print()
    print("  GENERATION:")
    print("    python -m critpt generate task_config=config.json model=openai/gpt-4")
    print()
    print("  EVALUATION:")
    print("    python -m critpt eval <file.json> --golden-data 'data/*.ipynb'")
    print("    python -m critpt eval-batch <dir> --golden-data 'data/*.ipynb'")
    print()
    print("  API SERVER:")
    print("    python -m critpt serve --port 8000 --golden-data 'data/*.ipynb'")
    print()
    print("=" * 70)
    print()
    print("Documentation:")
    print("  - README.md (overview)")
    print("  - EVALUATION_README.md (evaluation)")
    print("  - API_SERVER_SETUP.md (deployment)")
    print()


if __name__ == '__main__':
    main()
