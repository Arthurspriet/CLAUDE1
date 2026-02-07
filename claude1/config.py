"""Configuration constants and AppConfig dataclass."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude1.model_profiles import ModelProfile


# Default model for tool-calling tasks
DEFAULT_MODEL = "devstral-small-2:24b"

# Base directory for all claude1 data
DATA_DIR = Path.home() / ".claude1"
SESSIONS_DIR = DATA_DIR / "sessions"
MODELS_DIR = DATA_DIR / "models"
VIDEOS_DIR = DATA_DIR / "videos"
BANDITS_DIR = DATA_DIR / "bandits"
HISTORY_FILE = DATA_DIR / "history"

# Safety limits
MAX_TOOL_ITERATIONS = 25
DEFAULT_BASH_TIMEOUT = 120  # seconds
MAX_TOOL_OUTPUT_CHARS = 30_000
MAX_FILE_READ_CHARS = 50_000
MAX_GLOB_RESULTS = 200
MAX_GREP_MATCHES = 100

# ── Bash command safety ──────────────────────────────────────────────────────
# Blocked commands: hard reject, never allowed
BLOCKED_BASH_PATTERNS: list[str] = [
    r"rm\s+-[^\s]*r[^\s]*f[^\s]*\s+/\s*$",   # rm -rf /
    r"rm\s+-[^\s]*f[^\s]*r[^\s]*\s+/\s*$",   # rm -fr /
    r"mkfs\b",                                 # mkfs (format disk)
    r"dd\s+if=/dev/",                          # dd if=/dev/... (raw disk)
    r">\s*/dev/sd[a-z]",                       # > /dev/sda (overwrite disk)
    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",        # fork bomb :(){ :|:& };
    r"chmod\s+-R\s+777\s+/\s*$",              # chmod -R 777 /
    r"rm\s+-[^\s]*r[^\s]*\s+/\s*$",           # rm -r /
    r"mv\s+/\s+/dev/null",                     # mv / /dev/null
    r">\s*/dev/null\s*<\s*/dev/",              # redirect attacks
    r"wget\s+.*\|\s*(ba)?sh",                  # wget pipe to shell
    r"curl\s+.*\|\s*(ba)?sh",                  # curl pipe to shell
]

# Warned commands: require explicit confirmation even in auto-accept mode
WARNED_BASH_PATTERNS: list[str] = [
    r"rm\s+-[^\s]*r[^\s]*f",                  # rm -rf (any path)
    r"rm\s+-[^\s]*f[^\s]*r",                  # rm -fr (any path)
    r"git\s+reset\s+--hard",                  # git reset --hard
    r"git\s+push\s+.*--force",                # git push --force
    r"git\s+push\s+.*-f\b",                   # git push -f
    r"git\s+clean\s+-[^\s]*f",                # git clean -f
    r"chmod\s+-R\s+777",                       # chmod -R 777
    r"kill\s+-9",                              # kill -9
    r"killall\b",                              # killall
    r"pkill\b",                                # pkill
    r"shutdown\b",                             # shutdown
    r"reboot\b",                               # reboot
    r"systemctl\s+(stop|disable|mask)",        # systemctl stop/disable/mask
    r"docker\s+system\s+prune",               # docker system prune
    r"npm\s+publish",                          # npm publish
    r"pip\s+install\s+--force",               # pip install --force
]

# File path sandboxing
SENSITIVE_PATHS: list[str] = [
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.config/gcloud",
    "~/.kube",
    "~/.docker",
    "/etc/shadow",
    "/etc/passwd",
    "/etc/sudoers",
]

# Allowed paths outside working directory (always accessible)
ALLOWED_EXTRA_PATHS: list[str] = [
    "/tmp",
]

# Ollama settings
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = 8192

# HuggingFace settings
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")

# Solana / Jupiter settings
SOLANA_PRIVATE_KEY = os.environ.get("SOLANA_PRIVATE_KEY", "")
SOLANA_WALLET_ADDRESS = os.environ.get("SOLANA_WALLET_ADDRESS", "")
JUP_API_KEY = os.environ.get("JUP_API_KEY", "")
JUP_BASE_URL = "https://api.jup.ag"
JUP_HTTP_TIMEOUT = 30

# Well-known token mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
TOKEN_ALIASES: dict[str, str] = {"sol": SOL_MINT, "usdc": USDC_MINT, "usdt": USDT_MINT}

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# Timeouts (seconds)
HF_CLIENT_TIMEOUT = 120  # Overall HTTP timeout for HuggingFace InferenceClient
HF_STREAM_CHUNK_TIMEOUT = 90  # Max seconds to wait for a single stream chunk

# Context window management
CONTEXT_WINDOW_RESERVE = 2048

# Video generation settings
DEFAULT_VIDEO_MODEL_T2V = "damo-vilab/text-to-video-ms-1.7b"
DEFAULT_VIDEO_MODEL_I2V = "stabilityai/stable-video-diffusion-img2vid-xt"
DEFAULT_VIDEO_FRAMES = 25
DEFAULT_VIDEO_HEIGHT = 512
DEFAULT_VIDEO_WIDTH = 512
DEFAULT_VIDEO_FPS = 8


def _derive_base_model_name(repo_id: str) -> str:
    """Derive the base Ollama model name from an HF repo ID (without quant tag).
    
    This is similar to derive_model_name() in hf_import.py but without the quant suffix.
    
    Examples:
        'bartowski/Dolphin-3.0-Llama3.1-8B-GGUF' -> 'dolphin-3.0-llama3.1-8b'
        'meta-llama/Meta-Llama-3-8B-Instruct' -> 'meta-llama-3-8b-instruct'
    """
    # Get the model part (after org/)
    if "/" in repo_id:
        name = repo_id.split("/", 1)[1]
    else:
        name = repo_id
    
    # Strip common suffixes
    for suffix in ["-GGUF", "-gguf", "_GGUF", "_gguf"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    
    # Lowercase and clean
    name = name.lower().replace(" ", "-")
    
    # Remove any chars that Ollama doesn't like
    name = re.sub(r'[^a-z0-9._-]', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    
    return name


def check_local_ollama_model(hf_repo_id: str) -> str | None:
    """Check if an HF model exists locally as an Ollama model.
    
    Args:
        hf_repo_id: HuggingFace repo ID (e.g., "bartowski/Dolphin-3.0-Llama3.1-8B-GGUF")
    
    Returns:
        The matching Ollama model name if found, or None
    """
    try:
        import ollama
    except ImportError:
        return None
    
    try:
        # Get list of available Ollama models
        models_response = ollama.list()
        if not hasattr(models_response, 'models') or not models_response.models:
            return None
        
        available_models = [m.model for m in models_response.models if m.model]
        
        # Derive base name from HF repo ID
        base_name = _derive_base_model_name(hf_repo_id)
        
        # Check for exact matches or matches with quant tags
        # Models are typically named like "base-name:QUANT" or just "base-name"
        exact_match = None
        close_match = None
        
        for model_name in available_models:
            # Remove quant tag if present for comparison
            model_base = model_name.split(":")[0]
            
            # Check for exact match
            if model_base == base_name:
                exact_match = model_name
                break
            
            # Check for close matches (normalize by removing hyphens for comparison)
            # This handles cases like "dolphin3.0-llama3.1-8b" vs "dolphin-3.0-llama3.1-8b"
            base_normalized = re.sub(r'[-_]', '', base_name)
            model_normalized = re.sub(r'[-_]', '', model_base)
            
            if base_normalized == model_normalized:
                close_match = model_name
        
        # Return exact match if found, otherwise close match
        return exact_match or close_match
        
    except Exception:
        # If Ollama is not available or any error occurs, return None
        return None


def parse_model_spec(model: str) -> tuple[str, str]:
    """Parse a model spec into (provider, model_id).

    'hf:meta-llama/Meta-Llama-3-8B-Instruct' -> ('huggingface', 'meta-llama/Meta-Llama-3-8B-Instruct')
    'devstral-small-2:24b'                    -> ('ollama', 'devstral-small-2:24b')
    
    For HF models, checks if the model exists locally in Ollama first.
    If found locally, returns ('ollama', local_model_name) instead of using HuggingFace API.
    """
    if model.startswith("hf:"):
        hf_repo_id = model[3:]
        # Check if this model exists locally in Ollama
        local_model = check_local_ollama_model(hf_repo_id)
        if local_model:
            return ("ollama", local_model)
        # Fall back to HuggingFace API if not found locally
        return ("huggingface", hf_repo_id)
    return ("ollama", model)


# Config file path
CONFIG_FILE = DATA_DIR / "config.toml"


def load_config_file() -> dict:
    """Load settings from ~/.claude1/config.toml. Returns empty dict if not found."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        # Python 3.11+ has tomllib in stdlib
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        return tomllib.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


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
    auto_verify: bool = False
    bandit_enabled: bool = False
    model_info: dict | None = None
    profile: ModelProfile | None = None
    provider: str = "ollama"
    hf_endpoint: str = ""

    def __post_init__(self):
        # Ensure data directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        BANDITS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file_and_cli(cls, cli_overrides: dict) -> "AppConfig":
        """Create AppConfig by merging config file defaults with CLI overrides.

        Priority: CLI flags > config.toml > dataclass defaults
        """
        file_config = load_config_file()

        # Map TOML keys to AppConfig field names
        merged: dict = {}
        field_map = {
            "model": "model",
            "auto_accept_tools": "auto_accept_tools",
            "ollama_host": "ollama_host",
            "num_ctx": "num_ctx",
            "temperature": "temperature",
            "compact": "compact",
            "verbose": "verbose",
            "planning": "planning",
            "agents_mode": "agents_mode",
            "auto_verify": "auto_verify",
            "bandit_enabled": "bandit_enabled",
            "provider": "provider",
            "hf_endpoint": "hf_endpoint",
        }

        # Apply file config first
        for toml_key, field_name in field_map.items():
            if toml_key in file_config:
                merged[field_name] = file_config[toml_key]

        # CLI overrides take priority (only non-None/non-default values)
        for key, value in cli_overrides.items():
            if value is not None:
                merged[key] = value

        return cls(**merged)

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
