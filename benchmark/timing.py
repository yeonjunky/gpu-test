"""TTFT / end-to-end latency extraction from vLLM RequestOutput objects.

vLLM's RequestOutput.metrics attribute names have shifted across releases
(arrival_time / first_token_time / finished_time in some versions,
first_scheduled_time in others). This module reads them defensively via
getattr() so a version mismatch degrades to missing per-sample timing
(None) rather than crashing the whole combo run -- confirm the exact
attribute names for the pinned vLLM version in spike_tests/ and tighten
this if needed.
"""
from dataclasses import dataclass


@dataclass
class SampleTiming:
    ttft_ms: float | None
    e2e_ms: float | None
    output_tokens: int


def extract_timing(request_output) -> SampleTiming:
    output_tokens = sum(len(o.token_ids) for o in request_output.outputs)
    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return SampleTiming(ttft_ms=None, e2e_ms=None, output_tokens=output_tokens)

    arrival = getattr(metrics, "arrival_time", None)
    first_token = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)

    ttft_ms = (first_token - arrival) * 1000 if arrival is not None and first_token is not None else None
    e2e_ms = (finished - arrival) * 1000 if arrival is not None and finished is not None else None
    return SampleTiming(ttft_ms=ttft_ms, e2e_ms=e2e_ms, output_tokens=output_tokens)


def aggregate_task_perf(timings: list[SampleTiming], wall_clock_sec: float) -> dict:
    ttfts = [t.ttft_ms for t in timings if t.ttft_ms is not None]
    e2es = [t.e2e_ms for t in timings if t.e2e_ms is not None]
    total_output_tokens = sum(t.output_tokens for t in timings)
    return {
        "throughput_tok_per_sec": round(total_output_tokens / wall_clock_sec, 2) if wall_clock_sec > 0 else None,
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "avg_e2e_latency_ms": round(sum(e2es) / len(e2es), 2) if e2es else None,
        "n_samples": len(timings),
    }
