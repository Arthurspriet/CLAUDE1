"""prompt_toolkit input configuration."""

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

from config import HISTORY_FILE, DATA_DIR


def create_prompt_session() -> PromptSession:
    """Create a configured prompt_toolkit session."""
    # Ensure history file directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):
        """Escape+Enter inserts a newline."""
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def _(event):
        """Enter submits the input."""
        event.current_buffer.validate_and_handle()

    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        multiline=False,
        key_bindings=bindings,
        enable_history_search=True,
    )

    return session


def get_prompt_text(model_name: str) -> str:
    """Build the prompt string showing current model."""
    # Shorten the model name for display
    short = model_name.split(":")[0]
    return f"{short} > "
