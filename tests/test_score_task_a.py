from scorers import score_task_a


def make_expected(*calls):
    return {"function_calls": list(calls)}


def test_exact_match_scores_1():
    expected = make_expected(
        {"name": "get_weather", "args": {"city": ["Seoul"], "unit": ["celsius"]}},
    )
    output = '[{"name": "get_weather", "arguments": {"city": "Seoul", "unit": "celsius"}}]'
    result = score_task_a.score_sample(output, expected)
    assert result["valid_json"] is True
    assert result["score"] == 1.0
    assert result["n_matched_calls"] == 1


def test_missing_key_scores_partial():
    expected = make_expected({"name": "get_weather", "args": {"city": ["Seoul"], "unit": ["celsius"]}})
    output = '[{"name": "get_weather", "arguments": {"city": "Seoul"}}]'
    result = score_task_a.score_sample(output, expected)
    assert result["valid_json"] is True
    assert 0.0 < result["score"] < 1.0


def test_wrong_function_name_scores_0_for_that_call():
    expected = make_expected({"name": "get_weather", "args": {"city": ["Seoul"]}})
    output = '[{"name": "totally_different_function", "arguments": {"city": "Seoul"}}]'
    result = score_task_a.score_sample(output, expected)
    assert result["score"] == 0.0
    assert result["n_matched_calls"] == 0


def test_invalid_json_returns_0_without_crashing():
    expected = make_expected({"name": "get_weather", "args": {"city": ["Seoul"]}})
    result = score_task_a.score_sample("sorry, I can't help with that.", expected)
    assert result["valid_json"] is False
    assert result["score"] == 0.0


def test_json_wrapped_in_markdown_fence_is_extracted():
    expected = make_expected({"name": "get_weather", "args": {"city": ["Seoul"]}})
    output = '```json\n[{"name": "get_weather", "arguments": {"city": "Seoul"}}]\n```'
    result = score_task_a.score_sample(output, expected)
    assert result["valid_json"] is True
    assert result["score"] == 1.0


def test_type_mismatch_is_not_counted_as_match():
    expected = make_expected({"name": "set_count", "args": {"count": [5, 10]}})
    output = '[{"name": "set_count", "arguments": {"count": "five"}}]'
    result = score_task_a.score_sample(output, expected)
    assert result["score"] == 0.0


def test_multiple_calls_partial_credit():
    expected = make_expected(
        {"name": "fn_a", "args": {"x": [1]}},
        {"name": "fn_b", "args": {"y": [2]}},
    )
    output = '[{"name": "fn_a", "arguments": {"x": 1}}]'
    result = score_task_a.score_sample(output, expected)
    assert result["n_matched_calls"] == 1
    assert result["score"] == 0.5
