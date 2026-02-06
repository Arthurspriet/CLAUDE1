"""Configuration constants and AppConfig dataclass."""

import os
from dataclasses import dataclass, field
from pathlib import Path


# Default model for tool-calling tasks
DEFAULT_MODEL = "devstral-small-2:24b"

# Models known to support Ollama native tool calling
TOOL_CAPABLE_MODELS = {
    "devstral-small-2:24b",
    "devstral-small-2",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:1.7b",
    "cogito:14b",
    "cogito:8b",
    "mistral:instruct",
    "mistral:latest",
    "llama3.2:3b",
}

# Models known NOT to support tools
NON_TOOL_MODELS = {
    "gemma3",
    "deepseek-r1",
    "llama3.2:1b",
}

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

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# Context window management
CONTEXT_WINDOW_RESERVE = 2048


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
    model_info: dict | None = None

    def __post_init__(self):
        # Ensure data directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def model_short_name(self) -> str:
        """Return model name without tag for display."""
        return self.model.split(":")[0]
