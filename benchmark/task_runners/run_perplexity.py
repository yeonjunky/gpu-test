"""Perplexity runner (ablation study only, configs/ablation_matrix.yaml) --
NOT a generation task like task_a/b/c. Computes per-token NLL / perplexity
over fixed WikiText-2 chunks via vLLM's prompt_logprobs on the same already-
loaded model, rather than falling back to a separate `transformers` forward
pass: a transformers-based measurement would be blind to this study's own
kv_cache_dtype and fp8-activation-quant axes (neither has an equivalent in a
plain `model(input_ids, labels=input_ids)` call), while vLLM's prompt_logprobs
measures the actual served model, whatever axis combination is active.

GPU-only, like every other task_runner -- never exercised on the local
authoring machine. The exact shape of RequestOutput.prompt_logprobs (one
dict[token_id, Logprob] per prompt position, None at position 0 since there's
no preceding context) should be confirmed against the pinned vLLM version in
spike_tests/spike_test_build_llm_awq_gptq_fp8.py before trusting these numbers,
same defensive spirit as benchmark/timing.py's handling of vLLM's shifting
RequestOutput.metrics attribute names across releases.
"""
import json
import math
import time


def run(llm, data_path: str, task_cfg: dict) -> tuple[list[dict], dict, dict]:
    from vllm import SamplingParams

    chunks = [json.loads(line) for line in open(data_path)]
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)

    t0 = time.time()
    outputs = llm.generate([c["text"] for c in chunks], sampling_params)
    wall_clock = time.time() - t0

    raw_outputs = []
    total_nll, total_tokens = 0.0, 0
    for chunk, output in zip(chunks, outputs):
        prompt_token_ids = output.prompt_token_ids
        prompt_logprobs = output.prompt_logprobs
        # Position 0 has no preceding context (prompt_logprobs[0] is None) --
        # drop it, the standard WikiText2-perplexity convention.
        token_logprobs = [
            prompt_logprobs[i][prompt_token_ids[i]].logprob
            for i in range(1, len(prompt_token_ids))
            if prompt_logprobs[i] is not None
        ]
        sum_nll = -sum(token_logprobs)
        n_tokens = len(token_logprobs)
        total_nll += sum_nll
        total_tokens += n_tokens
        raw_outputs.append({"chunk_id": chunk["chunk_id"], "n_tokens": n_tokens, "sum_nll": round(sum_nll, 4)})

    avg_nll = total_nll / total_tokens if total_tokens else None
    perplexity = math.exp(avg_nll) if avg_nll is not None else None

    scores_summary = {
        "perplexity": round(perplexity, 4) if perplexity is not None else None,
        "avg_nll": round(avg_nll, 4) if avg_nll is not None else None,
        "n_chunks": len(chunks),
        "n_tokens": total_tokens,
    }
    perf_summary = {
        "wall_clock_sec": round(wall_clock, 2),
        "n_samples": len(chunks),
        "prefill_tok_per_sec": round(total_tokens / wall_clock, 2) if wall_clock > 0 else None,
    }
    return raw_outputs, scores_summary, perf_summary
