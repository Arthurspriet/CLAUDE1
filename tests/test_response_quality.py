"""Tests for response quality evaluation."""

from claude1.response_quality import (
    evaluate_response,
    estimate_response_quality,
    is_model_refusal,
    is_prompt_echo,
)


class TestRefusalDetection:
    """Test model refusal pattern matching."""

    def test_detects_refusal(self):
        assert is_model_refusal("I can't provide that information") is not None
        assert is_model_refusal("I cannot fulfill this request") is not None
        assert is_model_refusal("I'm unable to help with that") is not None
        assert is_model_refusal("As an AI, I can't do that") is not None

    def test_no_false_positive(self):
        assert is_model_refusal("Here is the implementation:") is None
        assert is_model_refusal("def hello(): pass") is None
        assert is_model_refusal("The file has been created successfully.") is None

    def test_empty_input(self):
        assert is_model_refusal("") is None
        assert is_model_refusal(None) is None  # type: ignore[arg-type]

    def test_refusal_in_first_300_chars(self):
        # Refusal after 300 chars should not be detected
        padding = "x" * 301
        assert is_model_refusal(padding + "I can't provide that") is None


class TestPromptEcho:
    """Test prompt echo detection."""

    def test_detects_echo(self):
        prompt = "Write a function that adds two numbers"
        output = "Write a function that adds two numbers"
        assert is_prompt_echo(output, prompt) is True

    def test_no_echo_for_different_content(self):
        prompt = "Write a function"
        output = "Here is the function:\ndef add(a, b): return a + b"
        assert is_prompt_echo(output, prompt) is False

    def test_empty_inputs(self):
        assert is_prompt_echo("", "prompt") is False
        assert is_prompt_echo("output", "") is False


class TestQualityScoring:
    """Test heuristic quality scoring."""

    def test_empty_response_scores_zero(self):
        assert estimate_response_quality("") == 0.0
        assert estimate_response_quality("   ") == 0.0

    def test_short_response_scores_low(self):
        score = estimate_response_quality("OK")
        assert score < 0.3

    def test_structured_response_scores_higher(self):
        response = (
            "## Overview\n\n"
            "Here is the implementation:\n\n"
            "```python\ndef hello():\n    pass\n```\n\n"
            "- Step 1: Create the function\n"
            "- Step 2: Add logic\n\n"
            "In conclusion, the implementation is complete."
        )
        score = estimate_response_quality(response)
        assert score >= 0.5


class TestEvaluateResponse:
    """Test combined evaluation."""

    def test_refusal_result(self):
        result = evaluate_response("I can't provide that")
        assert result.refusal is not None
        assert result.score == 0.0
        assert not result.is_acceptable

    def test_echo_result(self):
        prompt = "x" * 200
        result = evaluate_response(prompt, prompt)
        assert result.echo is True
        assert result.score == 0.0

    def test_good_response(self):
        response = (
            "## Solution\n\n"
            "Here's how to solve the problem:\n\n"
            "1. First, read the input\n"
            "2. Process the data\n"
            "3. Output the result\n\n"
            "```python\ndef solve(x):\n    return x * 2\n```\n\n"
            "In summary, this approach handles all edge cases."
        )
        result = evaluate_response(response, "How do I solve this?")
        assert result.is_acceptable is True
        assert result.score > 0.3
