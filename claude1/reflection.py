"""Reflection engine — post-execution analysis to improve future performance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReflectionResult:
    """Result of a reflection analysis."""

    insights: list[str] = field(default_factory=list)
    wasted_tool_calls: int = 0
    total_tool_calls: int = 0
    errors_encountered: int = 0
    retries: int = 0
    tool_effectiveness: dict[str, float] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ImprovementProposal:
    """A proposed self-improvement."""

    description: str
    target_file: str = ""
    change_type: str = ""  # "new_tool", "edit_tool", "new_skill", "config_change"
    confidence: float = 0.0


class ReflectionEngine:
    """Analyzes execution history and suggests improvements."""

    def reflect_on_execution(
        self,
        tool_calls: list[dict[str, Any]],
        total_duration: float = 0.0,
    ) -> ReflectionResult:
        """Analyze tool calls from a task execution.

        Identifies wasted calls, errors, retries, and tool effectiveness.
        """
        result = ReflectionResult()
        result.total_tool_calls = len(tool_calls)

        if not tool_calls:
            result.insights.append("No tool calls were made.")
            return result

        # Count errors and retries
        tool_names: list[str] = []
        for tc in tool_calls:
            tool_name = tc.get("tool", tc.get("tool_name", ""))
            tool_names.append(tool_name)
            result_text = tc.get("result_preview", tc.get("result", ""))

            if isinstance(result_text, str) and result_text.startswith("Error"):
                result.errors_encountered += 1

        # Detect wasted calls (same tool called with same args consecutively)
        for i in range(1, len(tool_calls)):
            if (tool_calls[i].get("tool") == tool_calls[i-1].get("tool") and
                    tool_calls[i].get("args") == tool_calls[i-1].get("args")):
                result.wasted_tool_calls += 1

        # Tool effectiveness (success rate per tool)
        tool_successes: dict[str, int] = {}
        tool_totals: dict[str, int] = {}
        for tc in tool_calls:
            name = tc.get("tool", tc.get("tool_name", "unknown"))
            tool_totals[name] = tool_totals.get(name, 0) + 1
            result_text = tc.get("result_preview", tc.get("result", ""))
            if not (isinstance(result_text, str) and result_text.startswith("Error")):
                tool_successes[name] = tool_successes.get(name, 0) + 1

        for name, total in tool_totals.items():
            successes = tool_successes.get(name, 0)
            result.tool_effectiveness[name] = successes / total if total > 0 else 0

        # Generate insights
        if result.wasted_tool_calls > 0:
            result.insights.append(
                f"{result.wasted_tool_calls} duplicate/wasted tool calls detected."
            )
            result.suggestions.append(
                "Avoid calling the same tool with the same arguments consecutively."
            )

        if result.errors_encountered > 0:
            error_rate = result.errors_encountered / result.total_tool_calls
            result.insights.append(
                f"{result.errors_encountered} errors ({error_rate:.0%} of calls)."
            )
            if error_rate > 0.3:
                result.suggestions.append(
                    "High error rate. Consider reading files before editing them."
                )

        # Check for read-before-edit pattern
        for i in range(1, len(tool_names)):
            if tool_names[i] in ("edit_file", "write_file") and tool_names[i-1] != "read_file":
                result.suggestions.append(
                    "Always read_file before edit_file to avoid errors."
                )
                break

        if total_duration > 0:
            result.insights.append(f"Total execution: {total_duration:.1f}s")

        return result

    def suggest_skill_from_patterns(
        self,
        recent_tool_calls: list[list[dict]],
    ) -> str | None:
        """If the same tool call pattern appears 3+ times across executions, suggest a skill.

        Args:
            recent_tool_calls: List of tool call lists from recent executions.

        Returns:
            A skill suggestion string, or None.
        """
        # Extract 2-grams of tool names from each execution
        pattern_counts: dict[str, int] = {}
        for calls in recent_tool_calls:
            names = [tc.get("tool", tc.get("tool_name", "")) for tc in calls]
            seen_in_exec: set[str] = set()
            for i in range(len(names) - 1):
                pair = f"{names[i]} -> {names[i+1]}"
                if pair not in seen_in_exec:
                    seen_in_exec.add(pair)
                    pattern_counts[pair] = pattern_counts.get(pair, 0) + 1

        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            if count >= 3:
                return (
                    f"Pattern '{pattern}' seen {count} times across executions. "
                    f"Consider creating a skill to automate this workflow."
                )

        return None

    def suggest_improvement(
        self,
        execution_log: list[dict],
    ) -> ImprovementProposal | None:
        """Identify limitations hit during execution and propose self-modifications."""
        # Look for repeated errors that suggest missing functionality
        error_messages: list[str] = []
        for entry in execution_log:
            result = entry.get("result_preview", entry.get("result", ""))
            if isinstance(result, str) and result.startswith("Error"):
                error_messages.append(result)

        if not error_messages:
            return None

        # Check for "unknown tool" errors
        for msg in error_messages:
            if "Unknown tool" in msg or "not available" in msg:
                return ImprovementProposal(
                    description=f"Missing tool referenced: {msg[:100]}",
                    change_type="new_tool",
                    confidence=0.8,
                )

        # Check for permission errors
        for msg in error_messages:
            if "Permission denied" in msg or "read-only" in msg.lower():
                return ImprovementProposal(
                    description="Permission issues detected. Consider adjusting tool permissions.",
                    change_type="config_change",
                    confidence=0.5,
                )

        return None
