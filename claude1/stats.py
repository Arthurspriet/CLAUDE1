"""Session statistics tracking for token usage and cost estimation.

Supports persistent stats saved to ~/.claude1/stats.json across sessions,
with per-model tracking.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from claude1.config import DATA_DIR

STATS_FILE = DATA_DIR / "stats.json"


@dataclass
class TurnStats:
    """Stats for a single LLM turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    eval_duration: int = 0  # nanoseconds
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.eval_duration > 0 and self.completion_tokens > 0:
            return self.completion_tokens / (self.eval_duration / 1e9)
        return 0.0


@dataclass
class AgentTurnStats:
    """Stats for a single agent run within multi-agent mode."""

    role: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class SessionStats:
    """Tracks token usage across a session."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    turns: list[TurnStats] = field(default_factory=list)
    agent_turns: list[AgentTurnStats] = field(default_factory=list)
    model_name: str = ""  # Current model for per-model tracking

    def add(self, prompt_tokens: int, completion_tokens: int, eval_duration: int = 0):
        """Record tokens from a single request."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1
        self.turns.append(TurnStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            eval_duration=eval_duration,
        ))

    def add_agent_run(self, role: str, model: str, prompt_tokens: int, completion_tokens: int):
        """Record tokens from an agent run in multi-agent mode."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1
        self.agent_turns.append(AgentTurnStats(
            role=role,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ))

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimate cost if using a paid API (input $0.003/1k, output $0.015/1k)."""
        return (self.total_prompt_tokens * 0.003 + self.total_completion_tokens * 0.015) / 1000

    def save_persistent(self) -> None:
        """Save current session stats to ~/.claude1/stats.json, merging with existing data."""
        history = load_persistent_stats()

        # Add current session entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.total_requests,
        }

        history["sessions"].append(entry)

        # Update per-model aggregates
        model_key = self.model_name or "unknown"
        if model_key not in history["models"]:
            history["models"][model_key] = {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_requests": 0,
                "session_count": 0,
            }
        m = history["models"][model_key]
        m["total_prompt_tokens"] += self.total_prompt_tokens
        m["total_completion_tokens"] += self.total_completion_tokens
        m["total_requests"] += self.total_requests
        m["session_count"] += 1

        # Update global totals
        history["total_prompt_tokens"] = history.get("total_prompt_tokens", 0) + self.total_prompt_tokens
        history["total_completion_tokens"] = history.get("total_completion_tokens", 0) + self.total_completion_tokens
        history["total_requests"] = history.get("total_requests", 0) + self.total_requests
        history["total_sessions"] = history.get("total_sessions", 0) + 1

        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            STATS_FILE.write_text(json.dumps(history, indent=2))
        except OSError:
            pass


def load_persistent_stats() -> dict:
    """Load persistent stats from disk."""
    if not STATS_FILE.exists():
        return {
            "sessions": [],
            "models": {},
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_requests": 0,
            "total_sessions": 0,
        }
    try:
        return json.loads(STATS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {
            "sessions": [],
            "models": {},
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_requests": 0,
            "total_sessions": 0,
        }
