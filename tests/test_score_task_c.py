from scorers import score_task_c


def test_exact_match_found():
    result = score_task_c.score_sample("The code is X7K2-PQ91-007.", "X7K2-PQ91-007")
    assert result["found"] is True
    assert result["score"] == 1.0


def test_not_found():
    result = score_task_c.score_sample("I don't know.", "X7K2-PQ91-007")
    assert result["found"] is False
    assert result["score"] == 0.0


def test_partial_string_is_not_a_match():
    result = score_task_c.score_sample("The code starts with X7K2 but I forget the rest.", "X7K2-PQ91-007")
    assert result["found"] is False
