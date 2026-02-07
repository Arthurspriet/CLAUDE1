"""Response quality evaluation for claude1.

Ported from deep_thinker's council refusal detection and quality evaluation.
Provides:
- Model refusal detection (pattern-matching on known refusal phrases)
- Prompt echo detection (model repeating the prompt instead of answering)
- Heuristic quality scoring (length, structure, completeness)
- Combined evaluation returning a structured result
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Refusal detection ────────────────────────────────────────────────────────

MODEL_REFUSAL_PATTERNS: list[str] = [
    "i can't provide",
    "i cannot provide",
    "i can't fulfill",
    "i cannot fulfill",
    "i'm unable to",
    "i am unable to",
    "i won't",
    "i will not",
    "i'm not able to",
    "i refuse to",
    "i can't help with",
    "i cannot help with",
    "can i help you with something else",
    "i'd be happy to help with something else",
    "i'm not comfortable",
    "i am not comfortable",
    "as an ai, i can't",
    "as an ai, i cannot",
    "i don't think i should",
    "i must respectfully decline",
]

# Models known to refuse analytical tasks (too small for nuanced perspective-taking)
REFUSAL_PRONE_MODELS: list[str] = [
    "llama3.2:1b",
    "qwen2.5:0.5b",
    "gemma:2b",
]


def is_model_refusal(text: str) -> str | None:
    """Check if model output is a refusal.

    Examines the first 300 characters (refusals appear at the start).

    Args:
        text: The model output text.

    Returns:
        The matched refusal pattern if found, None otherwise.
    """
    if not text:
        return None

    text_lower = text.lower()[:300]
    for pattern in MODEL_REFUSAL_PATTERNS:
        if pattern in text_lower:
            return pattern
    return None


def is_prompt_echo(output: str, prompt: str) -> bool:
    """Detect when a model echoes the prompt instead of answering.

    Args:
        output: The model output text.
        prompt: The original prompt sent to the model.

    Returns:
        True if the output appears to be an echo of the prompt.
    """
    if not output or not prompt:
        return False

    output_clean = output.strip().lower()
    prompt_clean = prompt.strip().lower()

    # Check if output starts with significant portion of the prompt
    prompt_signature = prompt_clean[:200]
    if prompt_signature and output_clean.startswith(prompt_signature[:100]):
        return True

    # Check for high overlap: if >60% of the output matches the prompt start
    if len(output_clean) > 50 and len(prompt_clean) > 50:
        # Compare character-by-character for the first min(len, 500) chars
        compare_len = min(len(output_clean), len(prompt_clean), 500)
        matches = sum(
            1 for a, b in zip(output_clean[:compare_len], prompt_clean[:compare_len])
            if a == b
        )
        if compare_len > 0 and matches / compare_len > 0.6:
            return True

    return False


# ── Quality scoring ──────────────────────────────────────────────────────────

# Minimum quality score to accept a response (0.0-1.0)
MIN_QUALITY_SCORE = 0.3
# Maximum retries for low-quality responses
MAX_QUALITY_RETRIES = 1
# Maximum retries for refusal responses
MAX_REFUSAL_RETRIES = 2
# Minimum output length (chars) to consider for quality evaluation
MIN_OUTPUT_LENGTH = 50

# Structure markers that indicate a well-organized response
_STRUCTURE_MARKERS = ["##", "**", "- ", "1.", "2.", "3.", "\n\n", "```"]

# Completeness markers that indicate a finished/concluded response
_COMPLETENESS_MARKERS = [
    "conclusion",
    "summary",
    "in summary",
    "overall",
    "recommendation",
    "therefore",
    "thus",
    "finally",
    "to summarize",
    "in conclusion",
]


def estimate_response_quality(text: str) -> float:
    """Estimate the quality of a model response using heuristics.

    Checks length, structure markers, and completeness indicators.

    Args:
        text: The response text to evaluate.

    Returns:
        Quality score between 0.0 and 1.0.
    """
    if not text or not text.strip():
        return 0.0

    text = text.strip()
    text_len = len(text)

    # Base score from length (0-0.4)
    if text_len < MIN_OUTPUT_LENGTH:
        length_score = 0.0
    elif text_len < 200:
        length_score = 0.15
    elif text_len < 500:
        length_score = 0.2
    elif text_len < 1000:
        length_score = 0.3
    elif text_len < 2000:
        length_score = 0.35
    else:
        length_score = 0.4

    # Structure score (0-0.3) — check for markdown, lists, sections
    markers_found = sum(1 for m in _STRUCTURE_MARKERS if m in text)
    structure_score = min(0.3, markers_found * 0.05)

    # Completeness score (0-0.3) — check for conclusion/summary indicators
    text_lower = text.lower()
    completeness_hits = sum(1 for m in _COMPLETENESS_MARKERS if m in text_lower)
    completeness_score = min(0.3, completeness_hits * 0.1)

    total = length_score + structure_score + completeness_score
    return min(1.0, total)


# ── Combined evaluation ──────────────────────────────────────────────────────


@dataclass
class ResponseQualityResult:
    """Structured result from response quality evaluation."""

    score: float = 0.0
    refusal: str | None = None
    echo: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """Whether the response is acceptable (no refusal/echo, score above minimum)."""
        return self.refusal is None and not self.echo and self.score >= MIN_QUALITY_SCORE

    @property
    def needs_retry(self) -> bool:
        """Whether this response should trigger a retry."""
        return self.refusal is not None or self.echo or self.score < MIN_QUALITY_SCORE

    def summary(self) -> str:
        """Human-readable summary of quality evaluation."""
        parts = [f"Quality: {self.score:.2f}"]
        if self.refusal:
            parts.append(f"Refusal: '{self.refusal}'")
        if self.echo:
            parts.append("Echo detected")
        if self.issues:
            parts.append(f"Issues: {', '.join(self.issues)}")
        return " | ".join(parts)


def evaluate_response(text: str, original_prompt: str = "") -> ResponseQualityResult:
    """Run all quality checks on a response.

    Args:
        text: The model response text.
        original_prompt: The original user prompt (for echo detection).

    Returns:
        ResponseQualityResult with all evaluation details.
    """
    result = ResponseQualityResult()

    # Check refusal
    result.refusal = is_model_refusal(text)
    if result.refusal:
        result.issues.append(f"Model refused: '{result.refusal}'")
        result.score = 0.0
        return result

    # Check echo
    result.echo = is_prompt_echo(text, original_prompt)
    if result.echo:
        result.issues.append("Model echoed prompt instead of answering")
        result.score = 0.0
        return result

    # Quality scoring
    result.score = estimate_response_quality(text)

    if result.score < MIN_QUALITY_SCORE:
        result.issues.append(f"Low quality score: {result.score:.2f} (min: {MIN_QUALITY_SCORE})")
    if len(text.strip()) < MIN_OUTPUT_LENGTH:
        result.issues.append(f"Response too short: {len(text.strip())} chars")

    return result

