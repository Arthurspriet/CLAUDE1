"""Configuration constants and AppConfig dataclass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_profiles import ModelProfile


# Default model for tool-calling tasks
DEFAULT_MODEL = "devstral-small-2:24b"

# Base directory for all claude1 data
DATA_DIR = Path.home() / ".claude1"
SESSIONS_DIR = DATA_DIR / "sessions"
HISTORY_FILE = DATA_DIR / "history"

# Safety limits
MAX_TOOL_ITERATIONS = 25
DEFAULT_BASH_TIMEOUT = 120  # seconds
MAX_TOOL_OUTPUT_CHARS = 30_000
MAX_FILE_READ_CHARS = 50_000
MAX_GLOB_RESULTS = 200
MAX_GREP_MATCHES = 100

# Ollama settings
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = 8192

# HuggingFace settings
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# Timeouts (seconds)
HF_CLIENT_TIMEOUT = 120  # Overall HTTP timeout for HuggingFace InferenceClient
HF_STREAM_CHUNK_TIMEOUT = 90  # Max seconds to wait for a single stream chunk

# Context window management
CONTEXT_WINDOW_RESERVE = 2048


def parse_model_spec(model: str) -> tuple[str, str]:
    """Parse a model spec into (provider, model_id).

    'hf:meta-llama/Meta-Llama-3-8B-Instruct' -> ('huggingface', 'meta-llama/Meta-Llama-3-8B-Instruct')
    'devstral-small-2:24b'                    -> ('ollama', 'devstral-small-2:24b')
    """
    if model.startswith("hf:"):
        return ("huggingface", model[3:])
    return ("ollama", model)


@dataclass
class AppConfig:
    """Runtime configuration for the application."""

    model: str = DEFAULT_MODEL
    working_dir: str = field(default_factory=lambda: os.getcwd())
    auto_accept_tools: bool = False
    ollama_host: str = OLLAMA_HOST
    num_ctx: int = NUM_CTX
    temperature: float | None = None
    compact: bool = False
    verbose: bool = False
    planning: bool = False
    agents_mode: bool = False
    model_info: dict | None = None
    profile: ModelProfile | None = None
    provider: str = "ollama"
    hf_endpoint: str = ""

    def __post_init__(self):
        # Ensure data directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def model_short_name(self) -> str:
        """Return model name without tag/prefix for display."""
        name = self.model
        # Strip hf: prefix
        if name.startswith("hf:"):
            name = name[3:]
        # Strip org/ prefix for HF models (e.g. 'meta-llama/Llama-3-8B' -> 'Llama-3-8B')
        if "/" in name:
            name = name.split("/", 1)[1]
        # Strip Ollama tag
        return name.split(":")[0]
