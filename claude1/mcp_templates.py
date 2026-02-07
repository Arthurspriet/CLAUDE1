"""Pre-built MCP server config templates for common services."""

from __future__ import annotations

# Each template is a dict that can be written directly to mcp.json
# under the "mcpServers" key.
MCP_TEMPLATES: dict[str, dict] = {
    "google-drive": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-google-drive"],
        "env": {
            "GOOGLE_APPLICATION_CREDENTIALS": "<path-to-service-account-key.json>",
        },
    },
    "gmail": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-gmail"],
        "env": {
            "GOOGLE_APPLICATION_CREDENTIALS": "<path-to-service-account-key.json>",
        },
    },
    "google-sheets": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-google-sheets"],
        "env": {
            "GOOGLE_APPLICATION_CREDENTIALS": "<path-to-service-account-key.json>",
        },
    },
    "google-calendar": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-google-calendar"],
        "env": {
            "GOOGLE_APPLICATION_CREDENTIALS": "<path-to-service-account-key.json>",
        },
    },
    "slack": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-slack"],
        "env": {
            "SLACK_BOT_TOKEN": "<your-slack-bot-token>",
        },
    },
    "github": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-github"],
        "env": {
            "GITHUB_TOKEN": "<your-github-token>",
        },
    },
}


def list_templates() -> list[dict]:
    """Return a list of available template summaries."""
    result = []
    for name, config in MCP_TEMPLATES.items():
        env_vars = list(config.get("env", {}).keys())
        result.append({
            "name": name,
            "command": config.get("command", ""),
            "args": config.get("args", []),
            "required_env": env_vars,
        })
    return result


def get_template(name: str) -> dict | None:
    """Get a template config by name. Returns None if not found."""
    return MCP_TEMPLATES.get(name)
