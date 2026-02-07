"""JSON persistence for execution plans and checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from claude1.config import DATA_DIR
from claude1.task_engine.types import ExecutionPlan


TASKS_DIR = DATA_DIR / "tasks"


def _ensure_dirs():
    TASKS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, data: dict):
    """Write JSON atomically using temp + rename."""
    _ensure_dirs()
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_execution(plan: ExecutionPlan) -> Path:
    """Save an execution plan to disk."""
    _ensure_dirs()
    path = TASKS_DIR / f"{plan.id}.json"
    _atomic_write(path, plan.to_dict())
    return path


def load_execution(plan_id: str) -> ExecutionPlan | None:
    """Load an execution plan from disk."""
    path = TASKS_DIR / f"{plan_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return ExecutionPlan.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def list_executions() -> list[dict]:
    """List all saved execution plans with summary info."""
    _ensure_dirs()
    results = []
    for f in sorted(TASKS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.name.startswith("checkpoint_"):
            continue
        try:
            data = json.loads(f.read_text())
            results.append({
                "id": data.get("id", f.stem),
                "description": data.get("description", ""),
                "status": data.get("status", "unknown"),
                "created_at": data.get("created_at", 0),
                "task_count": len(data.get("tasks", {})),
                "phase_count": len(data.get("phases", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return results


def save_checkpoint(plan: ExecutionPlan) -> Path:
    """Save a checkpoint of the execution plan."""
    _ensure_dirs()
    path = TASKS_DIR / f"checkpoint_{plan.id}.json"
    _atomic_write(path, plan.to_dict())
    return path


def load_checkpoint(plan_id: str) -> ExecutionPlan | None:
    """Load a checkpoint for an execution plan."""
    path = TASKS_DIR / f"checkpoint_{plan_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return ExecutionPlan.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
