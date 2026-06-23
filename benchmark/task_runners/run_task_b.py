"""Task B runner: code generation (LiveCodeBench) against a loaded vLLM model.

LiveCodeBench has two problem shapes that need different completion
instructions (see data/prep_scripts/prepare_task_b_livecodebench.py):
  - "functional" (LeetCode-style): model completes a `class Solution` method.
  - "stdin" (AtCoder-style): model writes a full stdin/stdout program.
"""
import json
import time

from benchmark.timing import aggregate_task_perf, extract_timing
from scorers import score_task_b

FUNCTIONAL_INSTRUCTIONS = (
    "Complete the following class so that the method passes all test cases. "
    "Respond with the COMPLETE, self-contained class definition (including "
    "`class Solution:` and any needed imports)."
)
STDIN_INSTRUCTIONS = (
    "Write a complete Python program that reads input from stdin and writes "
    "the answer to stdout, matching the I/O format described below exactly."
)


def _build_user_content(problem: dict) -> str:
    parts = [f"# {problem['question_title']} ({problem['difficulty']})\n\n{problem['question_content']}"]
    if problem["test_type"] == "functional":
        parts.append(f"\n{FUNCTIONAL_INSTRUCTIONS}\n\n```python\n{problem['starter_code']}\n```")
    else:
        parts.append(f"\n{STDIN_INSTRUCTIONS}")
    return "\n".join(parts)


def run(llm, data_path: str, task_cfg: dict) -> tuple[list[dict], dict, dict]:
    from vllm import SamplingParams

    problems = [json.loads(line) for line in open(data_path)]
    conversations = [
        [
            {"role": "system", "content": task_cfg["system_prompt"]},
            {"role": "user", "content": _build_user_content(p)},
        ]
        for p in problems
    ]

    sampling_params = SamplingParams(temperature=task_cfg["temperature"], max_tokens=task_cfg["max_tokens"])

    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params)
    wall_clock = time.time() - t0

    raw_outputs, timings, scores = [], [], []
    for problem, output in zip(problems, outputs):
        text = output.outputs[0].text
        timing = extract_timing(output)
        score = score_task_b.score_sample(text, problem)
        timings.append(timing)
        scores.append(score)
        raw_outputs.append({
            "id": problem["question_id"],
            "platform": problem["platform"],
            "difficulty": problem["difficulty"],
            "test_type": problem["test_type"],
            "model_output": text,
            "ttft_ms": timing.ttft_ms,
            "e2e_ms": timing.e2e_ms,
            "output_tokens": timing.output_tokens,
            "score": score,
        })

    accuracy = sum(s["score"] for s in scores) / len(scores) if scores else 0.0
    n_passed = sum(1 for s in scores if s.get("passed"))
    scores_summary = {"accuracy": round(accuracy, 4), "n_samples": len(scores), "n_passed": n_passed}
    perf_summary = aggregate_task_perf(timings, wall_clock)

    return raw_outputs, scores_summary, perf_summary
