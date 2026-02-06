"""Health check diagnostics for claude1."""

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import ollama

from config import AppConfig, DATA_DIR, SESSIONS_DIR


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    passed: bool
    message: str
    details: str = ""


def run_health_checks(config: AppConfig, tool_registry=None) -> list[CheckResult]:
    """Run all health checks and return results."""
    results: list[CheckResult] = []

    # 1. Ollama connectivity
    try:
        ollama.list()
        results.append(CheckResult("Ollama connectivity", True, "Connected"))
    except Exception as e:
        results.append(CheckResult("Ollama connectivity", False, "Cannot connect", str(e)))

    # 2. Current model availability
    try:
        models = ollama.list()
        names = [m.model for m in models.models if m.model]
        if config.model in names:
            results.append(CheckResult("Model available", True, f"{config.model} found"))
        else:
            # Check partial match (model name without tag)
            base = config.model.split(":")[0]
            matches = [n for n in names if n.startswith(base)]
            if matches:
                results.append(CheckResult("Model available", True, f"{config.model} found (as {matches[0]})"))
            else:
                results.append(CheckResult("Model available", False, f"{config.model} not found", f"Available: {', '.join(names[:5])}"))
    except Exception as e:
        results.append(CheckResult("Model available", False, "Cannot check", str(e)))

    # 3. Model responsiveness
    try:
        start = time.time()
        response = ollama.chat(
            model=config.model,
            messages=[{"role": "user", "content": "Say OK"}],
            stream=False,
            options={"num_ctx": 256},
        )
        elapsed = time.time() - start
        content = response.get("message", {}).get("content", "") if isinstance(response, dict) else getattr(getattr(response, "message", None), "content", "")
        if elapsed < 10:
            results.append(CheckResult("Model responsiveness", True, f"Responded in {elapsed:.1f}s", content[:100]))
        else:
            results.append(CheckResult("Model responsiveness", False, f"Slow response: {elapsed:.1f}s", content[:100]))
    except Exception as e:
        results.append(CheckResult("Model responsiveness", False, "No response", str(e)))

    # 4. Disk space
    try:
        usage = shutil.disk_usage(str(Path.home()))
        free_gb = usage.free / (1024 ** 3)
        if free_gb >= 1.0:
            results.append(CheckResult("Disk space", True, f"{free_gb:.1f} GB free"))
        else:
            results.append(CheckResult("Disk space", False, f"Low: {free_gb:.2f} GB free", "Recommend at least 1 GB free"))
    except Exception as e:
        results.append(CheckResult("Disk space", False, "Cannot check", str(e)))

    # 5. Config directory
    config_ok = DATA_DIR.is_dir()
    sessions_ok = SESSIONS_DIR.is_dir()
    if config_ok and sessions_ok:
        results.append(CheckResult("Config directory", True, str(DATA_DIR)))
    else:
        missing = []
        if not config_ok:
            missing.append(str(DATA_DIR))
        if not sessions_ok:
            missing.append(str(SESSIONS_DIR))
        results.append(CheckResult("Config directory", False, "Missing directories", ", ".join(missing)))

    # 6. CLAUDE.md presence (informational, always passes)
    claude_md_global = (DATA_DIR / "CLAUDE.md").exists()
    claude_md_project = (Path(config.working_dir) / "CLAUDE.md").exists()
    parts = []
    if claude_md_global:
        parts.append("global")
    if claude_md_project:
        parts.append("project")
    if parts:
        results.append(CheckResult("CLAUDE.md", True, f"Found: {', '.join(parts)}"))
    else:
        results.append(CheckResult("CLAUDE.md", True, "Not found (optional)", "Create ~/.claude1/CLAUDE.md or ./CLAUDE.md for custom instructions"))

    # 7. Tool registry health
    if tool_registry:
        tools = tool_registry.all_tools()
        valid = 0
        issues = []
        for tool in tools:
            try:
                defn = tool.to_ollama_tool()
                if defn.get("function", {}).get("name") and defn.get("function", {}).get("parameters"):
                    valid += 1
                else:
                    issues.append(f"{tool.name}: missing name or parameters")
            except Exception as e:
                issues.append(f"{tool.name}: {e}")
        if not issues:
            results.append(CheckResult("Tool registry", True, f"{valid} tools registered"))
        else:
            results.append(CheckResult("Tool registry", False, f"{valid}/{len(tools)} valid", "; ".join(issues)))
    else:
        results.append(CheckResult("Tool registry", True, "Not checked (no registry provided)"))

    # 8. Git availability
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip()
        results.append(CheckResult("Git", True, version))
    except FileNotFoundError:
        results.append(CheckResult("Git", False, "Not found", "Install git for version control features"))
    except Exception as e:
        results.append(CheckResult("Git", False, "Error", str(e)))

    # 9. Python version
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        results.append(CheckResult("Python version", True, version))
    else:
        results.append(CheckResult("Python version", False, version, "Python 3.10+ recommended"))

    # 10. Context window adequacy
    num_ctx = config.num_ctx
    if num_ctx >= 4096:
        results.append(CheckResult("Context window", True, f"{num_ctx} tokens"))
    else:
        results.append(CheckResult("Context window", False, f"{num_ctx} tokens", "Recommend at least 4096 for tool-calling"))

    return results
