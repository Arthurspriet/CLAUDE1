"""Structured logging and audit trail for claude1.

Provides:
- JSON file handler with rotation (~/.claude1/logs/)
- Dedicated audit log for tool executions
- Console handler respecting verbose mode
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

from claude1.config import DATA_DIR

LOGS_DIR = DATA_DIR / "logs"
AUDIT_LOG_FILE = LOGS_DIR / "audit.jsonl"
APP_LOG_FILE = LOGS_DIR / "claude1.log"

# Maximum log file size (5 MB) and backup count
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        # Include extra fields
        for key in ("tool_name", "tool_args", "tool_result", "duration_s", "model", "provider"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry)


def setup_logging(verbose: bool = False) -> None:
    """Configure application-wide logging.

    - File handler: JSON lines to ~/.claude1/logs/claude1.log (with rotation)
    - Console handler: only if verbose=True, WARNING+ level
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("claude1")
    root.setLevel(logging.DEBUG)

    # Remove existing handlers (idempotent)
    root.handlers.clear()

    # Rotating file handler (JSON)
    file_handler = logging.handlers.RotatingFileHandler(
        str(APP_LOG_FILE),
        maxBytes=MAX_LOG_BYTES,
        backupCount=LOG_BACKUP_COUNT,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    # Console handler (only in verbose mode)
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        root.addHandler(console_handler)


def get_audit_logger() -> logging.Logger:
    """Get the dedicated audit logger for tool executions."""
    logger = logging.getLogger("claude1.audit")
    # Add audit file handler if not already set up
    if not any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and str(AUDIT_LOG_FILE) in getattr(h, "baseFilename", "")
        for h in logger.handlers
    ):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            str(AUDIT_LOG_FILE),
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    return logger


def log_tool_execution(
    tool_name: str,
    tool_args: dict,
    result: str,
    duration_s: float,
) -> None:
    """Log a tool execution to the audit trail."""
    logger = get_audit_logger()
    # Truncate long results for the log
    result_preview = result[:500] if len(result) > 500 else result
    logger.info(
        "Tool executed: %s",
        tool_name,
        extra={
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": result_preview,
            "duration_s": round(duration_s, 3),
        },
    )
