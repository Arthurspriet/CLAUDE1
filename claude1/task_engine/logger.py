"""Persistent execution logger for tool calls and task completions."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude1.config import DATA_DIR


LOGS_DIR = DATA_DIR / "logs"


@dataclass
class ToolCallLog:
    """A logged tool call."""

    tool_name: str
    args_summary: str
    result_preview: str
    duration: float
    model: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args_summary": self.args_summary,
            "result_preview": self.result_preview,
            "duration": self.duration,
            "model": self.model,
            "timestamp": self.timestamp,
            "success": self.success,
        }


@dataclass
class ToolPattern:
    """A recurring tool call sequence."""

    sequence: list[str]
    count: int
    avg_duration: float = 0.0


class ExecutionLogger:
    """Persistent logger for tool calls, task completions, and patterns."""

    def __init__(self):
        self._tool_calls: list[ToolCallLog] = []
        self._log_file = LOGS_DIR / "tool_calls.jsonl"
        self._load()

    def _load(self):
        """Load existing log entries."""
        if not self._log_file.exists():
            return
        try:
            for line in self._log_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    self._tool_calls.append(ToolCallLog(
                        tool_name=data.get("tool_name", ""),
                        args_summary=data.get("args_summary", ""),
                        result_preview=data.get("result_preview", ""),
                        duration=data.get("duration", 0),
                        model=data.get("model", ""),
                        timestamp=data.get("timestamp", 0),
                        success=data.get("success", True),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue
        except OSError:
            pass

    def _append_to_log(self, entry: ToolCallLog):
        """Append a single entry to the log file."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def log_tool_call(self, tool_name: str, args: dict, result: str = "",
                      duration: float = 0.0, model: str = "",
                      success: bool = True):
        """Log a tool call."""
        args_summary = ", ".join(f"{k}={str(v)[:50]}" for k, v in args.items())
        entry = ToolCallLog(
            tool_name=tool_name,
            args_summary=args_summary[:200],
            result_preview=result[:200],
            duration=duration,
            model=model,
            success=success,
        )
        self._tool_calls.append(entry)
        self._append_to_log(entry)

    def log_task_completion(self, task_id: str, description: str,
                            status: str, duration: float = 0.0):
        """Log a task completion."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "task_completions.jsonl"
        entry = {
            "task_id": task_id,
            "description": description,
            "status": status,
            "duration": duration,
            "timestamp": time.time(),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def get_tool_analytics(self) -> dict[str, Any]:
        """Get tool usage analytics."""
        if not self._tool_calls:
            return {"total_calls": 0, "tools": {}}

        tool_counts = Counter(tc.tool_name for tc in self._tool_calls)
        tool_failures = Counter(
            tc.tool_name for tc in self._tool_calls if not tc.success
        )
        tool_durations: dict[str, list[float]] = {}
        for tc in self._tool_calls:
            if tc.tool_name not in tool_durations:
                tool_durations[tc.tool_name] = []
            tool_durations[tc.tool_name].append(tc.duration)

        tools_info: dict[str, dict] = {}
        for tool_name, count in tool_counts.most_common():
            durations = tool_durations.get(tool_name, [])
            avg_dur = sum(durations) / len(durations) if durations else 0
            failures = tool_failures.get(tool_name, 0)
            tools_info[tool_name] = {
                "count": count,
                "failures": failures,
                "failure_rate": failures / count if count > 0 else 0,
                "avg_duration": round(avg_dur, 3),
            }

        return {
            "total_calls": len(self._tool_calls),
            "unique_tools": len(tool_counts),
            "tools": tools_info,
        }

    def get_pattern_frequency(self, window: int = 2) -> list[ToolPattern]:
        """Find recurring tool call sequences."""
        if len(self._tool_calls) < window:
            return []

        sequences: dict[str, list[float]] = {}
        for i in range(len(self._tool_calls) - window + 1):
            seq = tuple(tc.tool_name for tc in self._tool_calls[i:i + window])
            key = " -> ".join(seq)
            duration = sum(tc.duration for tc in self._tool_calls[i:i + window])
            if key not in sequences:
                sequences[key] = []
            sequences[key].append(duration)

        patterns = []
        for seq_str, durations in sorted(sequences.items(), key=lambda x: -len(x[1])):
            if len(durations) >= 2:
                patterns.append(ToolPattern(
                    sequence=seq_str.split(" -> "),
                    count=len(durations),
                    avg_duration=sum(durations) / len(durations),
                ))

        return patterns[:20]  # Top 20 patterns

    def get_recent_calls(self, n: int = 20) -> list[ToolCallLog]:
        """Return the N most recent tool calls."""
        return self._tool_calls[-n:]
