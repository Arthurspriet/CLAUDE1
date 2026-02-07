"""Persistent, dependency-aware task engine.

Backwards-compatible with the old TaskManager API (create/complete/fail/list_tasks)
while adding phases, dependencies, checkpointing, and context management.
"""

from __future__ import annotations

import time
from typing import Any

from claude1.task_engine.types import TaskRecord, TaskState, PhaseRecord, ExecutionPlan
from claude1.task_engine import store as task_store


class TaskEngine:
    """Persistent task engine with dependency resolution and checkpointing.

    Drop-in replacement for TaskManager with added capabilities:
    - Phased execution plans
    - Dependency tracking (get_ready_tasks)
    - Subtask spawning
    - Shared context dict
    - Checkpoint / resume
    """

    def __init__(self):
        self._plan: ExecutionPlan | None = None
        self._tasks: dict[str, TaskRecord] = {}
        self._next_legacy_id = 1  # For backwards-compat integer IDs

    # ── Backwards-compatible TaskManager API ─────────────────────────────

    def create(self, description: str) -> TaskRecord:
        """Create and register a new task (TaskManager compat)."""
        task = TaskRecord(
            id=str(self._next_legacy_id),
            description=description,
            status=TaskState.RUNNING,
            started_at=time.time(),
        )
        self._next_legacy_id += 1
        self._tasks[task.id] = task
        if self._plan:
            self._plan.tasks[task.id] = task
        return task

    def complete(self, task_id: int | str, result: str):
        """Mark a task as completed (TaskManager compat)."""
        tid = str(task_id)
        task = self._tasks.get(tid)
        if task:
            task.status = TaskState.COMPLETED
            task.result = result
            task.completed_at = time.time()

    def fail(self, task_id: int | str, error: str):
        """Mark a task as failed (TaskManager compat)."""
        tid = str(task_id)
        task = self._tasks.get(tid)
        if task:
            task.status = TaskState.FAILED
            task.error = error
            task.result = error
            task.completed_at = time.time()

    def list_tasks(self) -> list[TaskRecord]:
        """Return all tasks (TaskManager compat)."""
        return list(self._tasks.values())

    # ── Execution plan management ────────────────────────────────────────

    def create_execution(self, description: str = "") -> ExecutionPlan:
        """Create a new execution plan."""
        plan = ExecutionPlan(description=description)
        plan.status = TaskState.RUNNING
        self._plan = plan
        return plan

    def get_execution(self) -> ExecutionPlan | None:
        """Return the current execution plan."""
        return self._plan

    def create_phase(self, name: str = "") -> PhaseRecord:
        """Create a new phase in the current execution plan."""
        if not self._plan:
            self.create_execution()
        phase = PhaseRecord(name=name)
        assert self._plan is not None
        self._plan.phases.append(phase)
        return phase

    def add_task_to_phase(self, phase_id: str, description: str,
                          depends_on: list[str] | None = None,
                          parent_id: str | None = None) -> TaskRecord:
        """Create a task and add it to a specific phase."""
        task = TaskRecord(
            description=description,
            depends_on=depends_on or [],
            parent_id=parent_id,
        )
        self._tasks[task.id] = task
        if self._plan:
            self._plan.tasks[task.id] = task
            # Find the phase and add the task
            for phase in self._plan.phases:
                if phase.id == phase_id:
                    phase.task_ids.append(task.id)
                    break
        return task

    def get_ready_tasks(self) -> list[TaskRecord]:
        """Return tasks whose dependencies are all resolved (completed)."""
        completed_ids = {
            tid for tid, t in self._tasks.items()
            if t.status == TaskState.COMPLETED
        }
        ready = []
        for task in self._tasks.values():
            if task.status != TaskState.PENDING:
                continue
            deps = task.depends_on
            if all(d in completed_ids for d in deps):
                ready.append(task)
        return ready

    def spawn_subtask(self, parent_id: str, description: str,
                      depends_on: list[str] | None = None) -> TaskRecord:
        """Spawn a subtask linked to a parent task."""
        task = TaskRecord(
            description=description,
            parent_id=parent_id,
            depends_on=depends_on or [],
        )
        self._tasks[task.id] = task
        if self._plan:
            self._plan.tasks[task.id] = task
        return task

    def start_task(self, task_id: str):
        """Mark a task as running."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskState.RUNNING
            task.started_at = time.time()

    def complete_task(self, task_id: str, result: str = "",
                      files_modified: list[str] | None = None,
                      token_usage: dict | None = None):
        """Complete a task with optional metadata."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskState.COMPLETED
            task.result = result
            task.completed_at = time.time()
            if files_modified:
                task.files_modified = files_modified
            if token_usage:
                task.token_usage = token_usage

    def fail_task(self, task_id: str, error: str):
        """Fail a task with an error message."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskState.FAILED
            task.error = error
            task.completed_at = time.time()

    def log_tool_call(self, task_id: str, tool_name: str, args: dict,
                      result: str = "", duration: float = 0.0):
        """Log a tool call against a task."""
        task = self._tasks.get(task_id)
        if task:
            task.tool_calls.append({
                "tool": tool_name,
                "args": {k: str(v)[:200] for k, v in args.items()},
                "result_preview": result[:200] if result else "",
                "duration": duration,
                "timestamp": time.time(),
            })

    # ── Context (shared state between tasks) ─────────────────────────────

    def set_context(self, key: str, value: Any):
        """Store a value in the shared execution context."""
        if self._plan:
            self._plan.shared_context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the shared execution context."""
        if self._plan:
            return self._plan.shared_context.get(key, default)
        return default

    # ── Persistence ──────────────────────────────────────────────────────

    def checkpoint(self):
        """Save the current state to disk."""
        if self._plan:
            task_store.save_checkpoint(self._plan)
            task_store.save_execution(self._plan)

    def save(self):
        """Persist the execution plan."""
        if self._plan:
            task_store.save_execution(self._plan)

    def resume(self, plan_id: str) -> bool:
        """Resume from a checkpoint or saved execution.

        Returns True if successfully loaded.
        """
        plan = task_store.load_checkpoint(plan_id)
        if plan is None:
            plan = task_store.load_execution(plan_id)
        if plan is None:
            return False

        self._plan = plan
        self._tasks = dict(plan.tasks)
        # Update legacy ID counter
        for tid in self._tasks:
            try:
                num = int(tid)
                if num >= self._next_legacy_id:
                    self._next_legacy_id = num + 1
            except ValueError:
                pass
        return True

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Look up a task by ID."""
        return self._tasks.get(task_id)

    @staticmethod
    def list_saved_executions() -> list[dict]:
        """List all saved execution plans."""
        return task_store.list_executions()

    def complete_execution(self):
        """Mark the current execution as completed and save."""
        if self._plan:
            self._plan.status = TaskState.COMPLETED
            self._plan.completed_at = time.time()
            self.save()
