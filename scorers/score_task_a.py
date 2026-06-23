"""Task A scorer: tool-use / JSON correctness.

Per the project spec: parse the model's output with json.loads(), compare
against the expected function call(s), and check that parameter keys and
value types match exactly (BFCL ground truth gives a list of acceptable
values per parameter -- a type match against ANY of those acceptable values
counts as correct, since BFCL itself allows multiple valid phrasings).

This is a deliberately simpler/stricter scorer than official BFCL AST-based
grading -- documented in the report methodology as a project-specific
simplification, not a reproduction of the official BFCL leaderboard score.
"""
import json
import re
from typing import Any

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_block(text: str) -> Any | None:
    """Pulls a JSON value out of model output text that may be wrapped in
    markdown fences or surrounded by prose. Returns None if nothing parses."""
    candidates = []
    fence_match = JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(text.strip())

    # also try slicing from the first '[' or '{' to the last ']' or '}'
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_calls(parsed: Any) -> list[dict] | None:
    """Model is instructed to output [{"name": ..., "arguments": {...}}, ...].
    Also accept a single dict (one call) for leniency."""
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None
    calls = []
    for item in parsed:
        if not isinstance(item, dict) or "name" not in item:
            continue
        args = item.get("arguments", item.get("args", {}))
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": item["name"], "arguments": args})
    return calls


def _type_matches(value: Any, acceptable_values: list[Any]) -> bool:
    for acceptable in acceptable_values:
        if isinstance(acceptable, bool):
            if isinstance(value, bool):
                return True
        elif isinstance(acceptable, (int, float)) and not isinstance(acceptable, bool):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
        elif isinstance(acceptable, str):
            if isinstance(value, str):
                return True
        elif isinstance(acceptable, list):
            if isinstance(value, list):
                return True
        elif isinstance(acceptable, dict):
            if isinstance(value, dict):
                return True
    return False


def _score_one_call(parsed_call: dict, expected_call: dict) -> dict:
    expected_args = expected_call["args"]
    parsed_args = parsed_call["arguments"]

    matched_keys = 0
    key_details = {}
    for key, acceptable_values in expected_args.items():
        if key not in parsed_args:
            key_details[key] = "missing"
            continue
        if _type_matches(parsed_args[key], acceptable_values):
            matched_keys += 1
            key_details[key] = "ok"
        else:
            key_details[key] = "type_mismatch"

    extra_keys = [k for k in parsed_args if k not in expected_args]
    n_expected = max(len(expected_args), 1)
    score = matched_keys / n_expected
    return {
        "score": score,
        "matched_keys": matched_keys,
        "n_expected_keys": len(expected_args),
        "extra_keys": extra_keys,
        "key_details": key_details,
    }


def score_sample(model_output_text: str, expected: dict) -> dict:
    expected_calls = expected["function_calls"]
    parsed = extract_json_block(model_output_text)
    if parsed is None:
        return {"valid_json": False, "score": 0.0, "n_expected_calls": len(expected_calls), "n_matched_calls": 0}

    parsed_calls = _normalize_calls(parsed)
    if parsed_calls is None:
        return {"valid_json": False, "score": 0.0, "n_expected_calls": len(expected_calls), "n_matched_calls": 0}

    used = [False] * len(parsed_calls)
    call_scores = []
    n_fully_matched = 0
    for expected_call in expected_calls:
        best_idx, best_detail = None, None
        for i, parsed_call in enumerate(parsed_calls):
            if used[i] or parsed_call["name"] != expected_call["name"]:
                continue
            detail = _score_one_call(parsed_call, expected_call)
            if best_detail is None or detail["score"] > best_detail["score"]:
                best_idx, best_detail = i, detail
        if best_detail is None:
            call_scores.append({"name": expected_call["name"], "score": 0.0, "reason": "no_matching_function_name"})
        else:
            used[best_idx] = True
            call_scores.append({"name": expected_call["name"], **best_detail})
            if best_detail["score"] == 1.0:
                n_fully_matched += 1

    overall_score = sum(c["score"] for c in call_scores) / max(len(expected_calls), 1)
    return {
        "valid_json": True,
        "score": round(overall_score, 4),
        "n_expected_calls": len(expected_calls),
        "n_matched_calls": n_fully_matched,
        "call_scores": call_scores,
    }
