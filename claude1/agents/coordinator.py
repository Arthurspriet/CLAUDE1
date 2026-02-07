"""Coordinator agent that decomposes user requests into task plans."""

from __future__ import annotations

import json
import re
from typing import Any

import ollama

from claude1.agents.types import AgentRole, TaskSpec, TaskResult, DecompositionPlan
from claude1.agents.config import DEFAULT_AGENT_CONFIGS


# Prompt that instructs the coordinator to produce a JSON decomposition
_DECOMPOSITION_PROMPT = """\
You are a task coordinator for a multi-agent coding assistant. Given a user request, \
decompose it into subtasks that can be executed by specialized agents.

Available agent roles:
- code_edit: Can read, write, edit files and run bash commands. Use for any task that modifies code.
- search: Can only read files and search the codebase. Use for finding information, reading code, listing files.
- reasoning: Has NO tools. Use for analysis, planning, or explaining concepts based on provided context.
- quick_lookup: Can only read files and search. Use for simple, fast lookups (small model).

Rules:
1. If the request is a single, simple task, create ONE task with the most appropriate role.
2. Tasks that modify the SAME files must NOT be parallel — use depends_on.
3. Search/read tasks can run in parallel with each other.
4. A code_edit task that needs search results should depend on the search task.
5. Keep task count minimal. Don't over-decompose simple requests.

Output ONLY this JSON structure (no markdown, no explanation):
{
  "summary": "Brief description of the plan",
  "tasks": [
    {
      "id": "task_1",
      "role": "code_edit|search|reasoning|quick_lookup",
      "instruction": "What the agent should do",
      "depends_on": []
    }
  ]
}
"""

_REPLAN_PROMPT = """\
You are a task coordinator for a multi-agent coding assistant. A previous plan partially failed \
and you need to create a revised plan for the REMAINING work.

Original request: {original_request}

Previous plan summary: {plan_summary}

Completed results:
{completed_summary}

Failure reason: {failure_reason}

Create a NEW plan for the remaining work, taking into account what was already completed \
and what went wrong. Avoid repeating tasks that already succeeded.

Available agent roles:
- code_edit: Can read, write, edit files and run bash commands.
- search: Can only read files and search the codebase.
- reasoning: Has NO tools. Analysis only.
- quick_lookup: Read-only, fast lookups.

Output ONLY this JSON structure (no markdown, no explanation):
{{
  "summary": "Brief description of the revised plan",
  "tasks": [
    {{
      "id": "task_1",
      "role": "code_edit|search|reasoning|quick_lookup",
      "instruction": "What the agent should do",
      "depends_on": []
    }}
  ]
}}
"""


class CoordinatorAgent:
    """Decomposes user requests into task plans and synthesizes multi-agent results."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        config_overrides: dict | None = None,
    ):
        self.client = ollama.AsyncClient(host=ollama_host)
        config = DEFAULT_AGENT_CONFIGS["coordinator"]
        if config_overrides and "coordinator" in config_overrides:
            config = config_overrides["coordinator"]
        self.model = config.model

    async def decompose(self, user_request: str) -> DecompositionPlan:
        """Decompose a user request into a structured plan.

        Falls back to a single code_edit task if JSON parsing fails.
        """
        messages = [
            {"role": "system", "content": _DECOMPOSITION_PROMPT},
            {"role": "user", "content": user_request},
        ]

        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options={"num_ctx": 4096},
            )

            content = response.get("message", {}).get("content", "") if isinstance(response, dict) else (response.message.content or "")
            plan = self._parse_plan(content)
            if plan:
                return plan

        except Exception:
            pass

        # Fallback: single code_edit task
        return self._fallback_plan(user_request)

    async def synthesize(
        self,
        original_request: str,
        plan: DecompositionPlan,
        results: dict[str, str],
    ) -> str:
        """Merge results from multiple agents into a coherent response."""
        parts = [
            f"Original request: {original_request}",
            f"Plan: {plan.summary}",
            "",
            "Results from agents:",
        ]
        for task in plan.tasks:
            output = results.get(task.id, "(no output)")
            parts.append(f"\n### {task.id} ({task.role.value}): {task.instruction}")
            parts.append(output)

        synthesis_prompt = (
            "You are synthesizing results from multiple agents that worked on parts of a user request. "
            "Combine their outputs into a single, coherent response. "
            "Highlight what was accomplished, any files changed, and any issues found."
        )

        messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": "\n".join(parts)},
        ]

        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options={"num_ctx": 4096},
            )
            content = response.get("message", {}).get("content", "") if isinstance(response, dict) else (response.message.content or "")
            return content.strip() if content else "Synthesis failed — see individual task results above."
        except Exception as e:
            return f"Synthesis error: {e}"

    async def replan(
        self,
        original_request: str,
        plan: DecompositionPlan,
        results: dict[str, TaskResult],
        failure_reason: str,
    ) -> DecompositionPlan:
        """Create a revised plan after partial failure.

        Takes the original request, current plan, results so far, and failure reason.
        Returns an updated DecompositionPlan for the remaining work.
        """
        # Build completed summary
        completed_parts = []
        for task in plan.tasks:
            result = results.get(task.id)
            if result and result.status.value == "completed":
                output_preview = result.output[:300] if result.output else "(no output)"
                completed_parts.append(f"- {task.id} ({task.role.value}): COMPLETED — {output_preview}")
            elif result and result.status.value == "failed":
                completed_parts.append(f"- {task.id} ({task.role.value}): FAILED — {result.error}")
            else:
                completed_parts.append(f"- {task.id} ({task.role.value}): NOT STARTED")

        prompt = _REPLAN_PROMPT.format(
            original_request=original_request,
            plan_summary=plan.summary,
            completed_summary="\n".join(completed_parts),
            failure_reason=failure_reason,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Create a revised plan."},
        ]

        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options={"num_ctx": 4096},
            )

            content = response.get("message", {}).get("content", "") if isinstance(response, dict) else (response.message.content or "")
            new_plan = self._parse_plan(content)
            if new_plan:
                return new_plan
        except Exception:
            pass

        # Fallback: single code_edit task with failure context
        return self._fallback_plan(
            f"{original_request}\n\nNote: Previous attempt failed: {failure_reason}"
        )

    def _parse_plan(self, raw: str) -> DecompositionPlan | None:
        """Parse JSON plan from model output. Handles markdown code blocks."""
        # Strip thinking tags (deepseek-r1 wraps in <think>...</think>)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()

        # Try to find raw JSON object
        if not raw.startswith("{"):
            brace_start = raw.find("{")
            if brace_start >= 0:
                raw = raw[brace_start:]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict) or "tasks" not in data:
            return None

        tasks = []
        valid_roles = {r.value for r in AgentRole if r != AgentRole.COORDINATOR}

        for t in data["tasks"]:
            role_str = t.get("role", "code_edit")
            if role_str not in valid_roles:
                role_str = "code_edit"

            tasks.append(TaskSpec(
                id=t.get("id", f"task_{len(tasks) + 1}"),
                role=AgentRole(role_str),
                instruction=t.get("instruction", ""),
                depends_on=t.get("depends_on", []),
            ))

        if not tasks:
            return None

        plan = DecompositionPlan(
            summary=data.get("summary", "Multi-task plan"),
            tasks=tasks,
            execution_order=self._build_execution_order(tasks),
        )
        return plan

    def _build_execution_order(self, tasks: list[TaskSpec]) -> list[list[str]]:
        """Build execution groups via topological sort.

        Returns a list of groups. Tasks within a group can run in parallel.
        Tasks in later groups depend on tasks in earlier groups.
        """
        task_ids = {t.id for t in tasks}
        task_map = {t.id: t for t in tasks}
        completed: set[str] = set()
        order: list[list[str]] = []

        remaining = set(task_ids)
        max_rounds = len(tasks) + 1  # prevent infinite loop

        for _ in range(max_rounds):
            if not remaining:
                break

            # Find tasks whose dependencies are all completed
            ready = []
            for tid in remaining:
                task = task_map[tid]
                deps = [d for d in task.depends_on if d in task_ids]
                if all(d in completed for d in deps):
                    ready.append(tid)

            if not ready:
                # Circular dependency — just run everything remaining
                ready = list(remaining)

            order.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return order

    def _fallback_plan(self, user_request: str) -> DecompositionPlan:
        """Create a single code_edit task as fallback."""
        task = TaskSpec(
            id="task_1",
            role=AgentRole.CODE_EDIT,
            instruction=user_request,
        )
        return DecompositionPlan(
            summary="Single-agent execution (fallback)",
            tasks=[task],
            execution_order=[["task_1"]],
        )
