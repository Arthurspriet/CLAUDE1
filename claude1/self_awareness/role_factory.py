"""Agent role factory — create and manage custom agent roles."""

from __future__ import annotations

import json
from pathlib import Path

from claude1.agents.config import AgentRoleConfig
from claude1.config import DATA_DIR


ROLES_FILE = DATA_DIR / "agent_roles.json"


class RoleFactory:
    """Create, update, and persist custom agent roles."""

    def __init__(self):
        self._custom_roles: dict[str, AgentRoleConfig] = {}
        self._load()

    def _load(self):
        """Load custom roles from disk."""
        if not ROLES_FILE.exists():
            return
        try:
            data = json.loads(ROLES_FILE.read_text())
            for name, role_data in data.items():
                self._custom_roles[name] = AgentRoleConfig(
                    model=role_data.get("model", ""),
                    tool_names=role_data.get("tool_names", []),
                    read_only=role_data.get("read_only", False),
                    max_iterations=role_data.get("max_iterations", 10),
                    system_prompt_extra=role_data.get("system_prompt_extra", ""),
                )
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self):
        """Persist custom roles to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, config in self._custom_roles.items():
            data[name] = {
                "model": config.model,
                "tool_names": config.tool_names,
                "read_only": config.read_only,
                "max_iterations": config.max_iterations,
                "system_prompt_extra": config.system_prompt_extra,
            }
        ROLES_FILE.write_text(json.dumps(data, indent=2))

    def create_role(
        self,
        name: str,
        model: str,
        tool_names: list[str] | None = None,
        read_only: bool = False,
        max_iterations: int = 10,
        system_prompt_extra: str = "",
    ) -> AgentRoleConfig:
        """Register a new custom agent role."""
        config = AgentRoleConfig(
            model=model,
            tool_names=tool_names or [],
            read_only=read_only,
            max_iterations=max_iterations,
            system_prompt_extra=system_prompt_extra,
        )
        self._custom_roles[name] = config
        self._save()

        # Also register in DEFAULT_AGENT_CONFIGS for immediate use
        try:
            from claude1.agents.config import DEFAULT_AGENT_CONFIGS
            DEFAULT_AGENT_CONFIGS[name] = config
        except Exception:
            pass

        return config

    def update_role(self, name: str, **overrides) -> AgentRoleConfig:
        """Update an existing custom role."""
        if name not in self._custom_roles:
            raise ValueError(f"Custom role '{name}' not found.")

        config = self._custom_roles[name]
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self._save()
        return config

    def list_custom_roles(self) -> list[AgentRoleConfig]:
        """Return all custom agent roles."""
        return list(self._custom_roles.values())

    def get_custom_role_names(self) -> list[str]:
        """Return names of all custom roles."""
        return list(self._custom_roles.keys())

    def delete_role(self, name: str) -> bool:
        """Delete a custom role."""
        if name in self._custom_roles:
            del self._custom_roles[name]
            self._save()
            return True
        return False
