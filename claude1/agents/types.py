"""Core types for the multi-agent system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    """Roles that agents can take in the hierarchy."""

    COORDINATOR = "coordinator"
    CODE_EDIT = "code_edit"
    SEARCH = "search"
    REASONING = "reasoning"
    QUICK_LOOKUP = "quick_lookup"


class TaskStatus(str, Enum):
    """Status of a task in the execution pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskSpec:
    """Specification for a single task to be executed by a worker agent."""

    id: str
    role: AgentRole
    instruction: str
    context: str = ""
    depends_on: list[str] = field(default_factory=list)
    structured_input: dict[str, Any] = field(default_factory=dict)
    output_schema: dict | None = None


@dataclass
class TaskResult:
    """Result of a completed task."""

    task_id: str
    role: AgentRole
    status: TaskStatus
    output: str = ""
    files_modified: list[str] = field(default_factory=list)
    error: str = ""
    token_usage: dict = field(default_factory=dict)
    structured_output: dict[str, Any] = field(default_factory=dict)
    spawned_tasks: list[TaskSpec] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class DecompositionPlan:
    """Plan produced by the coordinator for how to decompose a user request."""

    summary: str
    tasks: list[TaskSpec]
    execution_order: list[list[str]] = field(default_factory=list)
