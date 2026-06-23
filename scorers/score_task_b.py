"""Task B scorer: code generation (LiveCodeBench), sandboxed execution.

Two problem shapes, both graded via scorers/sandbox/exec_worker.py running in
an ISOLATED SUBPROCESS (never exec() in this, the main scoring process,
since the code under test is arbitrary LLM output):

  - "functional" (LeetCode-style): model is asked for a complete, self
    contained `class Solution` with the given method filled in. Test case
    inputs/outputs are parsed here (in the trusted parent process, from our
    own dataset, not model output) via ast.literal_eval into real Python
    values and handed to the worker as JSON.
  - "stdin" (AtCoder-style): model is asked for a full program reading from
    stdin / printing to stdout. Test case input/output strings are passed
    through as-is; the worker feeds each as stdin and compares captured
    stdout.

Only the dataset's public test cases are used (see prepare_task_b_livecodebench.py
docstring for why) -- typically 2-4 per problem, fewer than the official
LiveCodeBench private suite. Documented as a methodology simplification.
"""
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SANDBOX_DIR = Path(__file__).parent / "sandbox"
EXEC_WORKER = SANDBOX_DIR / "exec_worker.py"

TIMEOUT_SECONDS = 15  # a few test cases per problem, each capped individually by RLIMIT_CPU inside the worker
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
METHOD_NAME_RE = re.compile(r"def\s+(\w+)\s*\(\s*self")


def extract_code(model_output_text: str) -> str:
    match = CODE_FENCE_RE.search(model_output_text)
    if match:
        return match.group(1).strip()
    return model_output_text.strip()


def _extract_method_name(starter_code: str) -> str | None:
    match = METHOD_NAME_RE.search(starter_code)
    return match.group(1) if match else None


def _parse_functional_test_cases(public_tests: list[dict]) -> list[dict]:
    parsed = []
    for t in public_tests:
        arg_lines = [line for line in t["input"].split("\n") if line != ""]
        args = [ast.literal_eval(line) for line in arg_lines]
        expected = ast.literal_eval(t["output"])
        parsed.append({"args": args, "expected": expected})
    return parsed


# LeetCode-style starter_code signatures routinely use `List[int]`, `Optional[str]`,
# etc. Function annotations are evaluated eagerly at def-time, so a model that
# (correctly) solves the problem but omits `from typing import List` would
# otherwise fail with NameError before its logic even runs. Always prepending
# this is safe (no behavior change for code that already imports it).
TYPING_PRELUDE = "from typing import *\n\n"


def _build_spec(generated_code: str, problem: dict) -> dict:
    if problem["test_type"] == "functional":
        method_name = _extract_method_name(problem["starter_code"])
        if method_name is None:
            raise ValueError(f"could not extract method name from starter_code for {problem['question_id']}")
        return {
            "mode": "functional",
            "generated_code": TYPING_PRELUDE + generated_code,
            "method_name": method_name,
            "test_cases": _parse_functional_test_cases(problem["public_tests"]),
        }
    elif problem["test_type"] == "stdin":
        return {
            "mode": "stdin",
            "generated_code": TYPING_PRELUDE + generated_code,
            "test_cases": [{"input": t["input"], "output": t["output"]} for t in problem["public_tests"]],
        }
    else:
        raise ValueError(f"unknown test_type: {problem['test_type']}")


def score_sample(model_output_text: str, problem: dict) -> dict:
    """problem: one row from data/prepared/task_b_livecodebench_50.jsonl."""
    generated_code = extract_code(model_output_text)

    try:
        spec = _build_spec(generated_code, problem)
    except (ValueError, SyntaxError) as e:
        return {"passed": False, "score": 0.0, "returncode": None, "stderr_tail": f"spec build failed: {e}"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(spec, tmp)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, str(EXEC_WORKER), tmp_path],
            timeout=TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
            cwd=str(SANDBOX_DIR),
        )
        passed = result.returncode == 0
        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "score": 0.0,
            "returncode": None,
            "stderr_tail": f"TIMEOUT after {TIMEOUT_SECONDS}s",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
