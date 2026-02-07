"""Self-critique and refinement for claude1.

Ported from deep_thinker's CritiqueConsensus, adapted for single-model
self-critique. The model reviews its own output, identifies issues,
then produces a refined version incorporating the feedback.

Usage:
    result = run_self_critique(llm_interface, response_text, context)
    print(result.critique)
    print(result.refined_output)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from claude1.llm import LLMInterface

# Guarded import for huggingface_hub
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    InferenceClient = None
    HF_AVAILABLE = False


@dataclass
class CritiqueResult:
    """Result of self-critique and refinement."""

    original: str
    critique: str
    refined_output: str
    improvement_score: float  # Rough estimate of how much it improved (0-1)
    success: bool = True
    error: str = ""


# ── Prompt templates ─────────────────────────────────────────────────────────

_CRITIQUE_PROMPT = """You are a critical reviewer analyzing an AI-generated response.

{context_section}

## Response to Critique:
{response}

## Instructions:
Provide a constructive critique focusing on:
1. **Correctness** — Are there any errors or inaccuracies?
2. **Completeness** — Is anything important missing?
3. **Clarity** — Is the response clear and well-structured?
4. **Quality** — Could any aspects be improved?

Be specific and actionable. Focus on substantive issues, not minor stylistic preferences.
If the response is already excellent, say so and note only minor improvements.

## Critique:"""

_REFINEMENT_PROMPT = """You are refining an AI-generated response based on a self-review.

{context_section}

## Original Response:
{original}

## Critique:
{critique}

## Instructions:
1. Carefully consider each point in the critique
2. Address valid concerns and suggestions
3. Maintain the strengths of the original response
4. Produce an improved version that incorporates the feedback
5. Do NOT mention the critique in your output — just provide the refined content
6. If the original was already excellent, return it with only minor polish

## Refined Response:"""


def _build_context_section(context: str) -> str:
    """Build the optional context section for prompts."""
    if context and context.strip():
        return f"## Context:\n{context.strip()}"
    return ""


def _call_model(
    llm: LLMInterface,
    prompt: str,
) -> str:
    """Make a non-streaming call to the current model.

    Uses Ollama or HuggingFace depending on the LLM's provider config.

    Args:
        llm: The LLMInterface instance (for model/provider info).
        prompt: The prompt to send.

    Returns:
        Model response text.
    """
    messages = [{"role": "user", "content": prompt}]

    if llm.config.provider == "huggingface":
        client = llm._get_hf_client()
        response = client.chat_completion(
            model=llm.config.model[3:],  # strip hf: prefix
            messages=messages,
            max_tokens=4096,
            stream=False,
        )
        content = response.choices[0].message.content if response.choices else ""
        return content.strip() if content else ""
    else:
        response = ollama.chat(
            model=llm.config.model,
            messages=messages,
            stream=False,
            options={"num_ctx": llm.config.num_ctx},
        )
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            content = getattr(getattr(response, "message", None), "content", "")
        return content.strip() if content else ""


def generate_self_critique(
    llm: LLMInterface,
    response: str,
    context: str = "",
) -> str:
    """Generate a critique of a response using the current model.

    Args:
        llm: The LLMInterface instance.
        response: The response text to critique.
        context: Optional context about the original question.

    Returns:
        Critique text.
    """
    prompt = _CRITIQUE_PROMPT.format(
        context_section=_build_context_section(context),
        response=response,
    )
    try:
        return _call_model(llm, prompt)
    except Exception as e:
        return f"Unable to generate critique: {e}"


def refine_with_critique(
    llm: LLMInterface,
    original: str,
    critique: str,
    context: str = "",
) -> str:
    """Refine a response based on a critique.

    Args:
        llm: The LLMInterface instance.
        original: The original response text.
        critique: The critique to address.
        context: Optional context about the original question.

    Returns:
        Refined response text.
    """
    prompt = _REFINEMENT_PROMPT.format(
        context_section=_build_context_section(context),
        original=original,
        critique=critique,
    )
    try:
        return _call_model(llm, prompt)
    except Exception as e:
        return original  # Fall back to original on failure


def run_self_critique(
    llm: LLMInterface,
    response: str,
    context: str = "",
) -> CritiqueResult:
    """Run full self-critique pipeline: critique then refine.

    Args:
        llm: The LLMInterface instance.
        response: The response text to critique and refine.
        context: Optional context about the original question.

    Returns:
        CritiqueResult with critique, refined output, and metadata.
    """
    try:
        # Step 1: Generate critique
        critique = generate_self_critique(llm, response, context)
        if not critique or critique.startswith("Unable to"):
            return CritiqueResult(
                original=response,
                critique=critique or "No critique generated",
                refined_output=response,
                improvement_score=0.0,
                success=False,
                error=critique or "Critique generation failed",
            )

        # Step 2: Refine based on critique
        refined = refine_with_critique(llm, response, critique, context)

        # Step 3: Estimate improvement (rough heuristic)
        improvement = _estimate_improvement(response, refined)

        return CritiqueResult(
            original=response,
            critique=critique,
            refined_output=refined,
            improvement_score=improvement,
            success=True,
        )

    except Exception as e:
        return CritiqueResult(
            original=response,
            critique="",
            refined_output=response,
            improvement_score=0.0,
            success=False,
            error=str(e),
        )


def _estimate_improvement(original: str, refined: str) -> float:
    """Rough heuristic to estimate how much the refinement improved things.

    Compares length, structure, and content changes.

    Returns:
        Score from 0.0 (no change) to 1.0 (major improvement).
    """
    if not refined or refined == original:
        return 0.0

    score = 0.0

    # Length improvement (if refined is meaningfully longer)
    orig_len = len(original.strip())
    ref_len = len(refined.strip())
    if orig_len > 0:
        length_ratio = ref_len / orig_len
        if 1.1 < length_ratio < 3.0:
            score += 0.2  # Moderate expansion
        elif length_ratio >= 3.0:
            score += 0.1  # Too much expansion is suspicious

    # Structure improvement (more markdown markers in refined)
    structure_markers = ["##", "**", "- ", "1.", "```"]
    orig_markers = sum(1 for m in structure_markers if m in original)
    ref_markers = sum(1 for m in structure_markers if m in refined)
    if ref_markers > orig_markers:
        score += min(0.3, (ref_markers - orig_markers) * 0.1)

    # Content change (are they actually different?)
    # Use simple character-level difference ratio
    common = sum(1 for a, b in zip(original[:1000], refined[:1000]) if a == b)
    compare_len = min(len(original[:1000]), len(refined[:1000]))
    if compare_len > 0:
        similarity = common / compare_len
        if similarity < 0.5:
            score += 0.3  # Significant rewrite
        elif similarity < 0.8:
            score += 0.2  # Moderate changes
        elif similarity < 0.95:
            score += 0.1  # Minor changes

    return min(1.0, score)

