#!/usr/bin/env python3
"""Entrypoint for executing untrusted, LLM-generated code in an isolated
subprocess. NEVER import or call this from the main scoring process directly
-- it must always be launched via subprocess.run() from score_task_b.py so a
crash, infinite loop, or resource exhaustion in the generated code cannot
affect the parent process.

Reads a JSON spec describing which of two grading modes to run (LiveCodeBench
has two problem shapes -- see data/prep_scripts/prepare_task_b_livecodebench.py):

  "functional" (LeetCode-style): generated_code defines a `class Solution`;
  each test case calls a named method with parsed args and compares the
  return value to the expected value.

  "stdin" (AtCoder-style): generated_code is a full program; each test case
  feeds it a different stdin string and compares captured stdout.

Exit code 0 = all test cases passed. Non-zero = a test failed, the generated
code raised, or a resource limit was hit. Failure detail goes to stderr.

Usage: python exec_worker.py <path_to_spec.json>
"""
import io
import json
import sys

from resource_limits import apply_limits


def run_functional(generated_code: str, method_name: str, test_cases: list[dict]) -> tuple[bool, str]:
    namespace: dict = {}
    exec(compile(generated_code, "<generated>", "exec"), namespace)
    if "Solution" not in namespace:
        return False, "generated code does not define a `Solution` class"

    solution = namespace["Solution"]()
    method = getattr(solution, method_name, None)
    if method is None:
        return False, f"Solution has no method '{method_name}'"

    failures = []
    for i, tc in enumerate(test_cases):
        try:
            actual = method(*tc["args"])
        except Exception as e:
            failures.append(f"test {i}: raised {type(e).__name__}: {e}")
            continue
        if actual != tc["expected"]:
            failures.append(f"test {i}: expected {tc['expected']!r}, got {actual!r}")

    if failures:
        return False, "\n".join(failures[:5])
    return True, ""


def run_stdin(generated_code: str, test_cases: list[dict]) -> tuple[bool, str]:
    compiled = compile(generated_code, "<generated>", "exec")
    failures = []
    for i, tc in enumerate(test_cases):
        stdin_backup, stdout_backup = sys.stdin, sys.stdout
        sys.stdin = io.StringIO(tc["input"])
        sys.stdout = io.StringIO()
        try:
            exec(compiled, {"__name__": "__main__"})
            actual = sys.stdout.getvalue()
        except Exception as e:
            sys.stdin, sys.stdout = stdin_backup, stdout_backup
            failures.append(f"test {i}: raised {type(e).__name__}: {e}")
            continue
        sys.stdin, sys.stdout = stdin_backup, stdout_backup
        if actual.strip() != tc["output"].strip():
            failures.append(f"test {i}: expected {tc['output'].strip()!r}, got {actual.strip()!r}")

    if failures:
        return False, "\n".join(failures[:5])
    return True, ""


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: exec_worker.py <spec_json_path>", file=sys.stderr)
        sys.exit(2)

    apply_limits()

    with open(sys.argv[1]) as f:
        spec = json.load(f)

    mode = spec["mode"]
    if mode == "functional":
        passed, detail = run_functional(spec["generated_code"], spec["method_name"], spec["test_cases"])
    elif mode == "stdin":
        passed, detail = run_stdin(spec["generated_code"], spec["test_cases"])
    else:
        print(f"unknown mode: {mode}", file=sys.stderr)
        sys.exit(2)

    if not passed:
        print(detail, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
