"""
Submit answers for all 70 CritPt problems.

Edit MODEL and ANSWERS below — every problem_id Challenge_1_main..Challenge_70_main
must have an entry, or the script refuses to submit.
Burns one of your 10 daily submissions.

Usage:
    export AA_API_KEY=...
    python submit_placeholder.py
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import requests

sys.path.insert(0, "src")
from critpt.data_loader.notebook_loader import NotebookDataLoader

API_URL = "https://artificialanalysis.ai/api/v2/critpt/evaluate"

MODEL = "my-agent-v1"
GENERATION_CONFIG = {}


def _placeholder_for(i: int) -> str:
    """Use the challenge's own unfilled code template as the placeholder.

    Keeps the function signature the judge expects — so the call succeeds
    and we get a clean 'wrong answer' rather than a judge error.
    """
    nb = Path(f"data/public_test_challenges/Challenge_{i}.ipynb")
    template = NotebookDataLoader(nb).load_main_problem().code_template
    return f"```python\n{template}\n```"


# problem_id -> generated_code. Built from per-challenge templates.
ANSWERS: dict[str, str] = {
    f"Challenge_{i}_main": _placeholder_for(i) for i in range(1, 71)
}


def main() -> None:
    api_key = os.environ.get("AA_API_KEY") or sys.exit("Set AA_API_KEY")

    payload = {
        "submissions": [
            {
                "problem_id": f"Challenge_{i}_main",
                "generated_code": ANSWERS[f"Challenge_{i}_main"],
                "model": MODEL,
                "generation_config": GENERATION_CONFIG,
            }
            for i in range(1, 71)
        ],
        "batch_metadata": {
            "model": MODEL,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    print(f"Submitting {len(payload['submissions'])} answers as {MODEL!r}...")
    print("(The eval server is synchronous and slow — this can take several minutes.)")

    r = requests.post(
        API_URL,
        headers={"x-api-key": api_key},
        json=payload,
        timeout=60 * 60,
    )

    print(f"\nHTTP {r.status_code}")
    for h in ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"):
        if h in r.headers:
            print(f"{h}: {r.headers[h]}")

    try:
        body = r.json()
    except ValueError:
        print("\nNon-JSON response body:")
        print(r.text)
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("results/eval_responses")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODEL}_{ts}.json"
    out_path.write_text(json.dumps(body, indent=2))
    print(f"Saved response to {out_path}")

    print("\nResponse body:")
    print(json.dumps(body, indent=2))

    if not r.ok:
        sys.exit(1)

    acc = body.get("accuracy", 0.0)
    judge_errs = body.get("judge_error_count", 0)
    timeouts = body.get("server_timeout_count", 0)
    timeout_rate = body.get("timeout_rate", 0.0)
    print(f"\naccuracy:     {acc:.1%}")
    print(f"judge errors: {judge_errs} / 70")
    print(f"timeouts:     {timeouts} / 70 ({timeout_rate:.1%})")


if __name__ == "__main__":
    main()
