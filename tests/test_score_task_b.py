from scorers import score_task_b

FUNCTIONAL_PROBLEM = {
    "question_id": "toy_func",
    "test_type": "functional",
    "starter_code": "class Solution:\n    def addTwo(self, a: int, b: int) -> int:\n        ",
    "public_tests": [{"input": "2\n3", "output": "5"}, {"input": "10\n-2", "output": "8"}],
}

FUNCTIONAL_PROBLEM_WITH_TYPING = {
    "question_id": "toy_func_typing",
    "test_type": "functional",
    "starter_code": "class Solution:\n    def total(self, nums: list) -> int:\n        ",
    "public_tests": [{"input": "[1, 2, 3]", "output": "6"}],
}

STDIN_PROBLEM = {
    "question_id": "toy_stdin",
    "test_type": "stdin",
    "public_tests": [{"input": "2 3\n", "output": "5\n"}, {"input": "10 -2\n", "output": "8\n"}],
}


def test_functional_correct_solution_passes():
    output = "```python\nclass Solution:\n    def addTwo(self, a, b):\n        return a + b\n```"
    result = score_task_b.score_sample(output, FUNCTIONAL_PROBLEM)
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_functional_incorrect_solution_fails():
    output = "class Solution:\n    def addTwo(self, a, b):\n        return a - b\n"
    result = score_task_b.score_sample(output, FUNCTIONAL_PROBLEM)
    assert result["passed"] is False
    assert "expected" in result["stderr_tail"]


def test_functional_missing_typing_import_does_not_break_list_hint():
    # model omits "from typing import List" even though its signature uses it
    output = "class Solution:\n    def f(self, nums: List[int]) -> int:\n        return sum(nums)\n"
    problem = {
        "question_id": "toy_func_list_hint",
        "test_type": "functional",
        "starter_code": "class Solution:\n    def f(self, nums: List[int]) -> int:\n        ",
        "public_tests": [{"input": "[1, 2, 3]", "output": "6"}],
    }
    result = score_task_b.score_sample(output, problem)
    assert result["passed"] is True


def test_stdin_correct_solution_passes():
    output = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    result = score_task_b.score_sample(output, STDIN_PROBLEM)
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_stdin_incorrect_solution_fails():
    output = "a, b = map(int, input().split())\nprint(a - b)\n"
    result = score_task_b.score_sample(output, STDIN_PROBLEM)
    assert result["passed"] is False
    assert "expected" in result["stderr_tail"]


def test_stdin_infinite_loop_is_killed_not_hung():
    output = "while True:\n    pass\n"
    result = score_task_b.score_sample(output, STDIN_PROBLEM)
    assert result["passed"] is False


def test_functional_method_not_found_fails_gracefully():
    output = "class Solution:\n    def wrongName(self, a, b):\n        return a + b\n"
    result = score_task_b.score_sample(output, FUNCTIONAL_PROBLEM)
    assert result["passed"] is False
    assert "Solution has no method" in result["stderr_tail"]


def test_no_solution_class_fails_gracefully():
    output = "print('I refuse to answer')\n"
    result = score_task_b.score_sample(output, FUNCTIONAL_PROBLEM)
    assert result["passed"] is False
    assert "does not define a `Solution` class" in result["stderr_tail"]
