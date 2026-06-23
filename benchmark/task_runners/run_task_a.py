"""Task A runner: tool-use / JSON correctness against a loaded vLLM model."""
import json
import time

from benchmark.timing import aggregate_task_perf, extract_timing
from scorers import score_task_a


def run(llm, data_path: str, task_cfg: dict) -> tuple[list[dict], dict, dict]:
    from vllm import SamplingParams

    samples = [json.loads(line) for line in open(data_path)]
    conversations = []
    for s in samples:
        user_content = f"Available tools (JSON schemas):\n{json.dumps(s['tools'])}\n\nUser request: {s['question']}"
        conversations.append([
            {"role": "system", "content": task_cfg["system_prompt"]},
            {"role": "user", "content": user_content},
        ])

    sampling_params = SamplingParams(temperature=task_cfg["temperature"], max_tokens=task_cfg["max_tokens"])

    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params)
    wall_clock = time.time() - t0

    raw_outputs, timings, scores = [], [], []
    for sample, output in zip(samples, outputs):
        text = output.outputs[0].text
        timing = extract_timing(output)
        score = score_task_a.score_sample(text, sample["expected"])
        timings.append(timing)
        scores.append(score)
        raw_outputs.append({
            "id": sample["id"],
            "category": sample["category"],
            "model_output": text,
            "ttft_ms": timing.ttft_ms,
            "e2e_ms": timing.e2e_ms,
            "output_tokens": timing.output_tokens,
            "score": score,
        })

    accuracy = sum(s["score"] for s in scores) / len(scores) if scores else 0.0
    n_valid_json = sum(1 for s in scores if s.get("valid_json"))
    scores_summary = {"accuracy": round(accuracy, 4), "n_samples": len(scores), "n_valid_json": n_valid_json}
    perf_summary = aggregate_task_perf(timings, wall_clock)

    return raw_outputs, scores_summary, perf_summary
