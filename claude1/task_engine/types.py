"""Core types for the persistent task engine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    """Status of a task in the execution pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskRecord:
    """A persistent record of a task."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    status: TaskState = TaskState.PENDING
    parent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "depends_on": self.depends_on,
            "files_modified": self.files_modified,
            "tool_calls": self.tool_calls,
            "token_usage": self.token_usage,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            description=data.get("description", ""),
            status=TaskState(data.get("status", "pending")),
            parent_id=data.get("parent_id"),
            depends_on=data.get("depends_on", []),
            files_modified=data.get("files_modified", []),
            tool_calls=data.get("tool_calls", []),
            token_usage=data.get("token_usage", {}),
            result=data.get("result", ""),
            error=data.get("error", ""),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at", 0.0),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class PhaseRecord:
    """A phase in an execution plan (group of tasks)."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    task_ids: list[str] = field(default_factory=list)
    status: TaskState = TaskState.PENDING
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_ids": self.task_ids,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseRecord:
        return cls(
            id=data.get("id", uuid.uuid4().hex[:8]),
            name=data.get("name", ""),
            task_ids=data.get("task_ids", []),
            status=TaskState(data.get("status", "pending")),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at", 0.0),
        )


@dataclass
class ExecutionPlan:
    """A full execution plan with phases and shared context."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    phases: list[PhaseRecord] = field(default_factory=list)
    current_phase_index: int = 0
    shared_context: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    status: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "phases": [p.to_dict() for p in self.phases],
            "current_phase_index": self.current_phase_index,
            "shared_context": self.shared_context,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionPlan:
        tasks = {
            tid: TaskRecord.from_dict(tdata)
            for tid, tdata in data.get("tasks", {}).items()
        }
        phases = [PhaseRecord.from_dict(p) for p in data.get("phases", [])]
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            description=data.get("description", ""),
            phases=phases,
            current_phase_index=data.get("current_phase_index", 0),
            shared_context=data.get("shared_context", {}),
            tasks=tasks,
            status=TaskState(data.get("status", "pending")),
            created_at=data.get("created_at", time.time()),
            completed_at=data.get("completed_at", 0.0),
        )
