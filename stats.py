"""Session statistics tracking for token usage and cost estimation."""

from dataclasses import dataclass, field


@dataclass
class SessionStats:
    """Tracks token usage across a session."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int):
        """Record tokens from a single request."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimate cost if using a paid API (input $0.003/1k, output $0.015/1k)."""
        return (self.total_prompt_tokens * 0.003 + self.total_completion_tokens * 0.015) / 1000
