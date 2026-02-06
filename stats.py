"""Session statistics tracking for token usage and cost estimation."""

import time
from dataclasses import dataclass, field


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
