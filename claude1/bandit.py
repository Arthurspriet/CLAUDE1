"""Multi-armed bandit for model selection in claude1.

Simplified port of deep_thinker's GeneralizedBandit, focused on a single
use-case: learning which Ollama model produces the best responses over time.

Arms = installed Ollama model names.
Reward = response quality score (0.0-1.0) from response_quality.py.

Supports UCB1 and Thompson Sampling algorithms.
State is persisted to ~/.claude1/bandits/model_selection.json.

Usage:
    bandit = get_model_bandit()         # singleton, auto-discovers models
    model  = bandit.select()            # pick the best model
    bandit.update("qwen2.5:7b", 0.82)  # record quality score
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude1.config import BANDITS_DIR

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

STATE_FILE = BANDITS_DIR / "model_selection.json"


@dataclass
class BanditConfig:
    """Configuration for the model selection bandit."""

    enabled: bool = True
    algorithm: str = "ucb"  # "ucb" or "thompson"
    min_trials: int = 3  # Minimum pulls per arm before exploiting
    exploration_bonus: float = 1.0  # UCB exploration coefficient (c)


# ── Arm ──────────────────────────────────────────────────────────────────────


@dataclass
class BanditArm:
    """State for a single bandit arm (model)."""

    name: str
    pulls: int = 0
    total_reward: float = 0.0
    total_reward_sq: float = 0.0
    last_pull: str | None = None

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls

    @property
    def variance(self) -> float:
        if self.pulls < 2:
            return 1.0  # high uncertainty
        mean = self.mean_reward
        return max(0.01, (self.total_reward_sq / self.pulls) - (mean * mean))

    def ucb_score(self, total_pulls: int, exploration_bonus: float = 1.0) -> float:
        """UCB1 score: mean + c * sqrt(log(n) / n_i)."""
        if self.pulls == 0:
            return float("inf")  # unexplored arms get priority
        exploitation = self.mean_reward
        exploration = exploration_bonus * math.sqrt(
            math.log(total_pulls + 1) / self.pulls
        )
        return exploitation + exploration

    def thompson_sample(self) -> float:
        """Sample from Normal posterior (Thompson Sampling)."""
        if self.pulls == 0:
            return random.gauss(0.5, 0.5)  # uninformative prior centred at 0.5
        mean = self.mean_reward
        std = math.sqrt(self.variance / max(self.pulls, 1))
        return random.gauss(mean, std)

    def record(self, reward: float) -> None:
        """Record a reward observation."""
        self.pulls += 1
        self.total_reward += reward
        self.total_reward_sq += reward * reward
        self.last_pull = datetime.now(timezone.utc).isoformat()

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pulls": self.pulls,
            "total_reward": self.total_reward,
            "total_reward_sq": self.total_reward_sq,
            "last_pull": self.last_pull,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BanditArm:
        return cls(
            name=data.get("name", ""),
            pulls=data.get("pulls", 0),
            total_reward=data.get("total_reward", 0.0),
            total_reward_sq=data.get("total_reward_sq", 0.0),
            last_pull=data.get("last_pull"),
        )


# ── ModelBandit ──────────────────────────────────────────────────────────────


class ModelBandit:
    """Multi-armed bandit for Ollama model selection.

    Arms are automatically populated from Ollama's local model list.
    State is persisted across sessions.
    """

    def __init__(
        self,
        config: BanditConfig | None = None,
        state_path: Path | None = None,
    ):
        self.config = config or BanditConfig()
        self.state_path = state_path or STATE_FILE
        self.arms: dict[str, BanditArm] = {}
        self.total_pulls: int = 0
        self._load_state()

    # ── Public API ───────────────────────────────────────────────────────

    def select(self) -> str:
        """Select the best model using the configured algorithm.

        Returns:
            Model name, or empty string if no arms available.
        """
        self._ensure_arms()
        if not self.arms:
            return ""

        # Exploration phase: round-robin under-explored arms
        if self.total_pulls < self.config.min_trials * len(self.arms):
            for arm in self.arms.values():
                if arm.pulls < self.config.min_trials:
                    return arm.name

        # Exploitation
        if self.config.algorithm == "thompson":
            return self._thompson_select()
        return self._ucb_select()

    def update(self, model: str, reward: float) -> None:
        """Record a reward observation for a model.

        If the model isn't tracked yet, it's added as a new arm.

        Args:
            model: Ollama model name.
            reward: Quality score (0.0-1.0).
        """
        if model not in self.arms:
            self.arms[model] = BanditArm(name=model)

        self.arms[model].record(reward)
        self.total_pulls += 1
        self._save_state()

    def reset(self) -> None:
        """Clear all learned state."""
        self.arms.clear()
        self.total_pulls = 0
        self._save_state()
        logger.info("[BANDIT] State reset")

    def get_stats(self) -> dict[str, Any]:
        """Return human-readable statistics."""
        if not self.arms:
            return {"initialized": False}

        return {
            "initialized": True,
            "total_pulls": self.total_pulls,
            "exploration_complete": (
                self.total_pulls >= self.config.min_trials * len(self.arms)
            ),
            "best_arm": max(
                self.arms.values(), key=lambda a: a.mean_reward
            ).name
            if self.arms
            else None,
            "arms": {
                name: {
                    "pulls": arm.pulls,
                    "mean_reward": arm.mean_reward,
                    "ucb_score": arm.ucb_score(
                        self.total_pulls, self.config.exploration_bonus
                    ),
                }
                for name, arm in self.arms.items()
            },
        }

    # ── Selection algorithms ─────────────────────────────────────────────

    def _ucb_select(self) -> str:
        best_arm = max(
            self.arms.values(),
            key=lambda a: a.ucb_score(self.total_pulls, self.config.exploration_bonus),
        )
        return best_arm.name

    def _thompson_select(self) -> str:
        best_arm = max(self.arms.values(), key=lambda a: a.thompson_sample())
        return best_arm.name

    # ── Arm discovery ────────────────────────────────────────────────────

    def _ensure_arms(self) -> None:
        """Populate arms from installed Ollama models if empty."""
        if self.arms:
            return
        try:
            import ollama

            models = ollama.list()
            names = [m.model for m in models.models if m.model]
            for name in sorted(names):
                if name not in self.arms:
                    self.arms[name] = BanditArm(name=name)
        except Exception as e:
            logger.warning(f"[BANDIT] Failed to list Ollama models: {e}")

    # ── Persistence ──────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                self.total_pulls = data.get("total_pulls", 0)
                for arm_data in data.get("arms", []):
                    arm = BanditArm.from_dict(arm_data)
                    if arm.name:
                        self.arms[arm.name] = arm
                logger.info(f"[BANDIT] Loaded state: {self.total_pulls} pulls, {len(self.arms)} arms")
        except Exception as e:
            logger.warning(f"[BANDIT] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_pulls": self.total_pulls,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "arms": [arm.to_dict() for arm in self.arms.values()],
            }
            self.state_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"[BANDIT] Failed to save state: {e}")


# ── Singleton access ─────────────────────────────────────────────────────────

_model_bandit: ModelBandit | None = None


def get_model_bandit() -> ModelBandit:
    """Return the singleton ModelBandit instance."""
    global _model_bandit
    if _model_bandit is None:
        _model_bandit = ModelBandit()
    return _model_bandit

