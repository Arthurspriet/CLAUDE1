"""Bash command execution tool."""

import subprocess
from typing import Any

from config import DEFAULT_BASH_TIMEOUT, MAX_TOOL_OUTPUT_CHARS
from tools.base import BaseTool


class BashTool(BaseTool):
    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute a bash command in the working directory. "
            f"Output is truncated at {MAX_TOOL_OUTPUT_CHARS} characters. "
            f"Default timeout: {DEFAULT_BASH_TIMEOUT}s (override with timeout parameter). "
            "Use for: running scripts, git, package managers, builds, tests. "
            "Prefer grep_search over 'bash grep' and glob_search over 'bash find'. "
            "SAFETY: Do not run destructive commands (rm -rf, git reset --hard) without user approval. "
            "If a command fails, read stderr and adjust — don't blindly retry."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default: {DEFAULT_BASH_TIMEOUT})",
                },
            },
            "required": ["command"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", DEFAULT_BASH_TIMEOUT)

        if not command:
            return "Error: command is required"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.working_dir),
            )

            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if result.returncode != 0:
                output = f"[exit code: {result.returncode}]\n{output}"

            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                output = output[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [truncated at {MAX_TOOL_OUTPUT_CHARS} chars]"

            return output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {e}"
