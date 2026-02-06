"""Core types for the multi-agent system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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


@dataclass
class DecompositionPlan:
    """Plan produced by the coordinator for how to decompose a user request."""

    summary: str
    tasks: list[TaskSpec]
    execution_order: list[list[str]] = field(default_factory=list)
