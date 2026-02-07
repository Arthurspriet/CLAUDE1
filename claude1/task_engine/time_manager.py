"""Wall-clock time budgeting for long-running tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TaskTimeManager:
    """Manages wall-clock time budgets for task execution."""

    total_budget_seconds: float = 0  # 0 means unlimited
    reserve_seconds: float = 30.0  # reserved for final synthesis
    grace_period_seconds: float = 10.0  # extra time before hard abort

    _start_time: float = 0.0
    _phase_start_time: float = 0.0
    _phase_name: str = ""
    _active: bool = False

    def start(self):
        """Start the time budget clock."""
        self._start_time = time.time()
        self._active = True

    def start_phase(self, name: str = ""):
        """Mark the beginning of a phase."""
        self._phase_start_time = time.time()
        self._phase_name = name

    def end_phase(self):
        """Mark the end of the current phase."""
        self._phase_name = ""
        self._phase_start_time = 0.0

    def wall_remaining(self) -> float:
        """Return seconds remaining in the budget (inf if unlimited)."""
        if not self._active or self.total_budget_seconds <= 0:
            return float("inf")
        elapsed = time.time() - self._start_time
        return max(0.0, self.total_budget_seconds - elapsed)

    def should_abort_phase(self) -> bool:
        """Return True if the current phase should be aborted (time low)."""
        remaining = self.wall_remaining()
        if remaining == float("inf"):
            return False
        return remaining <= self.reserve_seconds

    def is_expired(self) -> bool:
        """Return True if the full budget (including grace) is exhausted."""
        remaining = self.wall_remaining()
        if remaining == float("inf"):
            return False
        return remaining <= -self.grace_period_seconds

    def get_status(self) -> dict:
        """Return a status dict for display."""
        elapsed = time.time() - self._start_time if self._active else 0.0
        remaining = self.wall_remaining()
        phase_elapsed = time.time() - self._phase_start_time if self._phase_start_time else 0.0

        return {
            "active": self._active,
            "elapsed_seconds": elapsed,
            "remaining_seconds": remaining if remaining != float("inf") else None,
            "budget_seconds": self.total_budget_seconds or None,
            "reserve_seconds": self.reserve_seconds,
            "current_phase": self._phase_name,
            "phase_elapsed_seconds": phase_elapsed,
            "should_wrap_up": self.should_abort_phase(),
        }
