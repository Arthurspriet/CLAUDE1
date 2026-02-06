"""Agent orchestrator — decomposes tasks, runs workers in parallel, synthesizes results."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from agents.types import TaskStatus, DecompositionPlan, TaskResult
from agents.config import AgentRoleConfig, DEFAULT_AGENT_CONFIGS
from agents.coordinator import CoordinatorAgent
from agents.worker import WorkerAgent
from agents.file_lock import FileLockManager
from tools import ToolRegistry


class OrchestratorEventType(str, Enum):
    """Types of events emitted during orchestration."""

    PLAN = "plan"
    AGENT_START = "agent_start"
    AGENT_DONE = "agent_done"
    SYNTHESIS = "synthesis"
    ERROR = "error"
    DONE = "done"


@dataclass
class OrchestratorEvent:
    """An event emitted by the orchestrator for UI rendering."""

    type: OrchestratorEventType
    task_id: str = ""
    role: str = ""
    model: str = ""
    status: str = ""
    content: str = ""
    plan: DecompositionPlan | None = None
    result: TaskResult | None = None


class AgentOrchestrator:
    """Main entry point for multi-agent execution.

    Flow:
    1. CoordinatorAgent decomposes the user request into a plan
    2. Workers execute tasks in parallel groups (respecting dependencies)
    3. If multiple tasks: CoordinatorAgent synthesizes results
    4. Events are emitted for UI rendering
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        working_dir: str,
        ollama_host: str = "http://localhost:11434",
        config_overrides: dict[str, AgentRoleConfig] | None = None,
    ):
        self.tool_registry = tool_registry
        self.working_dir = working_dir
        self.ollama_host = ollama_host
        self.config_overrides = config_overrides or {}
        self.file_lock = FileLockManager()
        self.coordinator = CoordinatorAgent(
            ollama_host=ollama_host,
            config_overrides=config_overrides,
        )
        self._active_workers: list[WorkerAgent] = []
        self._cancelled = False

    def cancel(self):
        """Cancel all active workers."""
        self._cancelled = True
        for worker in self._active_workers:
            worker.cancel()

    async def run(self, user_request: str) -> tuple[str, list[OrchestratorEvent]]:
        """Execute a user request through the multi-agent pipeline.

        Returns (final_output, events).
        """
        self._cancelled = False
        events: list[OrchestratorEvent] = []

        # Step 1: Decompose
        plan = await self.coordinator.decompose(user_request)
        events.append(OrchestratorEvent(
            type=OrchestratorEventType.PLAN,
            content=plan.summary,
            plan=plan,
        ))

        if self._cancelled:
            return "[cancelled]", events

        # Step 2: Execute task groups
        results: dict[str, TaskResult] = {}

        for group in plan.execution_order:
            if self._cancelled:
                break

            # Spawn workers for this parallel group
            tasks = []
            for task_id in group:
                task_spec = next((t for t in plan.tasks if t.id == task_id), None)
                if task_spec is None:
                    continue

                # Inject dependency results as context
                if task_spec.depends_on:
                    dep_outputs = []
                    for dep_id in task_spec.depends_on:
                        dep_result = results.get(dep_id)
                        if dep_result and dep_result.output:
                            dep_outputs.append(f"[{dep_id}]: {dep_result.output}")
                    if dep_outputs:
                        task_spec.context = "\n\n".join(dep_outputs)

                # Resolve config
                role_config = self.config_overrides.get(
                    task_spec.role.value,
                    DEFAULT_AGENT_CONFIGS.get(task_spec.role.value),
                )

                worker = WorkerAgent(
                    role=task_spec.role,
                    tool_registry=self.tool_registry,
                    file_lock=self.file_lock,
                    working_dir=self.working_dir,
                    role_config=role_config,
                    ollama_host=self.ollama_host,
                )
                self._active_workers.append(worker)

                events.append(OrchestratorEvent(
                    type=OrchestratorEventType.AGENT_START,
                    task_id=task_spec.id,
                    role=task_spec.role.value,
                    model=role_config.model if role_config else "unknown",
                    status="running",
                    content=task_spec.instruction,
                ))

                tasks.append((task_spec, worker))

            # Run group in parallel
            if tasks:
                coros = [worker.execute(spec) for spec, worker in tasks]
                group_results = await asyncio.gather(*coros, return_exceptions=True)

                for (task_spec, worker), result in zip(tasks, group_results):
                    if isinstance(result, Exception):
                        task_result = TaskResult(
                            task_id=task_spec.id,
                            role=task_spec.role,
                            status=TaskStatus.FAILED,
                            error=str(result),
                        )
                    else:
                        task_result = result

                    results[task_spec.id] = task_result
                    events.append(OrchestratorEvent(
                        type=OrchestratorEventType.AGENT_DONE,
                        task_id=task_spec.id,
                        role=task_spec.role.value,
                        status=task_result.status.value,
                        content=task_result.output[:500] if task_result.output else task_result.error,
                        result=task_result,
                    ))

                # Clean up workers
                for _, worker in tasks:
                    if worker in self._active_workers:
                        self._active_workers.remove(worker)

        # Step 3: Produce final output
        if self._cancelled:
            return "[cancelled]", events

        completed_results = {
            tid: r.output for tid, r in results.items()
            if r.status == TaskStatus.COMPLETED and r.output
        }

        if len(plan.tasks) == 1:
            # Single task — return its output directly
            single_result = results.get(plan.tasks[0].id)
            if single_result:
                final = single_result.output or single_result.error or "(no output)"
            else:
                final = "(no result)"
        else:
            # Multiple tasks — synthesize
            final = await self.coordinator.synthesize(
                user_request, plan, completed_results,
            )
            events.append(OrchestratorEvent(
                type=OrchestratorEventType.SYNTHESIS,
                content=final[:200],
            ))

        events.append(OrchestratorEvent(
            type=OrchestratorEventType.DONE,
            content=final[:200],
        ))

        return final, events
