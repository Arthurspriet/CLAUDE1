"""Task tracking for spawned subtasks."""

import time
from dataclasses import dataclass, field


@dataclass
class TaskRecord:
    """A record of a spawned subtask."""

    id: int
    description: str
    status: str = "running"  # "running", "completed", "failed"
    result: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


class TaskManager:
    """Tracks spawned subtasks for the /tasks command."""

    def __init__(self):
        self.tasks: list[TaskRecord] = []
        self._next_id = 1

    def create(self, description: str) -> TaskRecord:
        """Create and register a new task."""
        task = TaskRecord(id=self._next_id, description=description)
        self._next_id += 1
        self.tasks.append(task)
        return task

    def complete(self, task_id: int, result: str):
        """Mark a task as completed."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = "completed"
                task.result = result
                task.completed_at = time.time()
                return

    def fail(self, task_id: int, error: str):
        """Mark a task as failed."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = "failed"
                task.result = error
                task.completed_at = time.time()
                return

    def list_tasks(self) -> list[TaskRecord]:
        """Return all tasks."""
        return list(self.tasks)
