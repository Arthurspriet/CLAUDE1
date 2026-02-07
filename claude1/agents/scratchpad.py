"""Shared inter-agent scratchpad for real-time data sharing."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScratchpadEntry:
    """A single entry in the scratchpad with history tracking."""

    value: Any
    agent_id: str
    timestamp: float = field(default_factory=time.time)


class SharedScratchpad:
    """Thread-safe shared scratchpad for inter-agent communication.

    Allows agents in a parallel group to share data in real time.
    Tracks write history per key for auditability.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._history: dict[str, list[ScratchpadEntry]] = {}
        self._lock = threading.Lock()

    def write(self, key: str, value: Any, agent_id: str = ""):
        """Write a value to the scratchpad."""
        with self._lock:
            self._data[key] = value
            entry = ScratchpadEntry(value=value, agent_id=agent_id)
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(entry)

    def read(self, key: str, default: Any = None) -> Any:
        """Read a value from the scratchpad."""
        with self._lock:
            return self._data.get(key, default)

    def list_keys(self) -> list[str]:
        """List all keys in the scratchpad."""
        with self._lock:
            return list(self._data.keys())

    def get_history(self, key: str) -> list[ScratchpadEntry]:
        """Get the write history for a key."""
        with self._lock:
            return list(self._history.get(key, []))

    def to_dict(self) -> dict[str, Any]:
        """Serialize scratchpad state for persistence."""
        with self._lock:
            return {
                "data": {k: str(v)[:1000] for k, v in self._data.items()},
                "history": {
                    k: [
                        {"value": str(e.value)[:200], "agent_id": e.agent_id, "timestamp": e.timestamp}
                        for e in entries
                    ]
                    for k, entries in self._history.items()
                },
            }

    def clear(self):
        """Clear all data."""
        with self._lock:
            self._data.clear()
            self._history.clear()
