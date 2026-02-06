"""Main REPL orchestration - connects input, LLM, and display."""

import json
import signal
import sys

import ollama
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from config import AppConfig
from model_profiles import get_profile, format_profile_info
from llm import LLMInterface, StreamChunk
from session import save_session, load_session, list_sessions, auto_save_session, get_latest_session, export_as_markdown
from stats import SessionStats
from tools import ToolRegistry
from ui.prompts import create_prompt_session, get_prompt_text
from ui import renderer

console = Console()


class REPL:
    """Interactive REPL that orchestrates input -> LLM -> display."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.auto_accept = config.auto_accept_tools
        self.tool_registry = ToolRegistry(config.working_dir)
        self.llm = LLMInterface(
            config=config,
            tool_registry=self.tool_registry,
            confirm_callback=self._confirm_tool,
        )
        self.prompt_session = create_prompt_session()
        self._running = True
        self.session_stats = SessionStats()

    def _confirm_tool(self, tool_name: str, tool_args: dict) -> bool:
        """Confirmation callback invoked by the LLM before executing dangerous tools."""
        if self.auto_accept:
            renderer.show_tool_call(tool_name, tool_args)
            return True

        approved, always = renderer.confirm_tool(tool_name, tool_args)
        if always:
            self.auto_accept = True
            renderer.show_info("Auto-accept enabled for this session.")
        return approved

    def _handle_slash_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            renderer.show_help()
            return True

        elif cmd == "/model":
            if not arg:
                renderer.show_info(f"Current model: {self.config.model}")
                renderer.show_info("Usage: /model <name>")
                return True
            self.config.model = arg
            # Resolve profile for new model
            ollama_family = None
            try:
                info = ollama.show(arg)
                details = info.get("details", {}) if isinstance(info, dict) else getattr(info, "details", None)
                if details:
                    ollama_family = details.get("family", "") if isinstance(details, dict) else getattr(details, "family", "")
            except Exception:
                pass
            profile = get_profile(arg, ollama_family=ollama_family)
            self.config.profile = profile
            self.config.num_ctx = profile.num_ctx
            self.llm.update_model(arg)
            renderer.show_model_changed(arg)
            renderer.show_info(format_profile_info(profile))
            if not profile.supports_tools:
                renderer.show_info("Tools disabled for this model — text-only mode.")
            return True

        elif cmd == "/profile":
            if self.config.profile:
                renderer.show_info(format_profile_info(self.config.profile))
            else:
                renderer.show_info("No profile loaded.")
            return True

        elif cmd == "/models":
            try:
                models = ollama.list()
                names = [m.model for m in models.models]
                renderer.show_models(sorted(names), self.config.model)
            except Exception as e:
                renderer.show_error(f"Failed to list models: {e}")
            return True

        elif cmd == "/clear":
            self.llm.clear_history()
            renderer.show_info("Conversation history cleared.")
            return True

        elif cmd == "/save":
            name = arg or ""
            if not name:
                renderer.show_info("Usage: /save <session_name>")
                return True
            path = save_session(self.llm.messages, name)
            renderer.show_info(f"Session saved: {path}")
            return True

        elif cmd == "/load":
            name = arg or ""
            if not name:
                renderer.show_info("Usage: /load <session_name>")
                return True
            messages = load_session(name)
            if messages is None:
                renderer.show_error(f"Session not found: {name}")
            else:
                self.llm.messages = messages
                renderer.show_info(f"Session loaded: {name} ({len(messages)} messages)")
            return True

        elif cmd == "/sessions":
            sessions = list_sessions()
            if not sessions:
                renderer.show_info("No saved sessions.")
            else:
                for s in sessions:
                    renderer.show_info(f"  {s['name']}  ({s['messages']} msgs, saved {s['saved_at']})")
            return True

        elif cmd == "/resume":
            messages = get_latest_session()
            if messages is None:
                renderer.show_error("No autosaved session found.")
            else:
                self.llm.messages = messages
                renderer.show_info(f"Resumed session ({len(messages)} messages)")
            return True

        elif cmd == "/auto":
            self.auto_accept = not self.auto_accept
            state = "ON" if self.auto_accept else "OFF"
            renderer.show_info(f"Auto-accept tools: {state}")
            return True

        elif cmd == "/temp":
            if not arg:
                current = self.config.temperature
                if current is None:
                    renderer.show_info("Temperature: default (model decides)")
                else:
                    renderer.show_info(f"Temperature: {current}")
                renderer.show_info("Usage: /temp <0.0-2.0> or /temp reset")
                return True
            if arg.strip().lower() == "reset":
                self.config.temperature = None
                renderer.show_info("Temperature reset to default.")
                return True
            try:
                val = float(arg)
                if not 0.0 <= val <= 2.0:
                    renderer.show_error("Temperature must be between 0.0 and 2.0")
                    return True
                self.config.temperature = val
                renderer.show_info(f"Temperature set to {val}")
            except ValueError:
                renderer.show_error("Invalid temperature value. Use a number between 0.0 and 2.0")
            return True

        elif cmd == "/compact":
            self.config.compact = not self.config.compact
            self.llm._set_system_prompt()
            state = "ON" if self.config.compact else "OFF"
            renderer.show_info(f"Compact mode: {state}")
            return True

        elif cmd == "/plan":
            self.config.planning = not self.config.planning
            state = "ON" if self.config.planning else "OFF"
            renderer.show_info(f"Planning mode: {state}")
            self.llm._set_system_prompt()
            return True

        elif cmd == "/undo":
            result = self.tool_registry.undo_stack.undo_last()
            renderer.show_info(result)
            return True

        elif cmd == "/export":
            filepath = arg.strip() if arg.strip() else None
            try:
                path = export_as_markdown(self.llm.messages, filepath)
                renderer.show_info(f"Conversation exported to: {path}")
            except Exception as e:
                renderer.show_error(f"Export failed: {e}")
            return True

        elif cmd in ("/exit", "/quit"):
            self._running = False
            return True

        return False

    def _process_response(self, chunks):
        """Process streaming chunks from the LLM and render them."""
        text_buffer = ""
        live = None
        spinner = None
        turn_prompt = 0
        turn_completion = 0
        turn_eval_duration = 0

        try:
            for chunk in chunks:
                if chunk.type == "thinking_start":
                    spinner = renderer.start_thinking_spinner()

                elif chunk.type == "thinking_stop":
                    renderer.stop_thinking_spinner(spinner)
                    spinner = None

                elif chunk.type == "text":
                    text_buffer += chunk.content
                    # Update live display
                    if live is None:
                        live = Live(
                            Markdown(text_buffer),
                            console=console,
                            refresh_per_second=10,
                            vertical_overflow="visible",
                        )
                        live.start()
                    else:
                        live.update(Markdown(text_buffer))

                elif chunk.type == "plan":
                    if live is not None:
                        live.stop()
                        live = None
                        text_buffer = ""
                    renderer.show_plan(chunk.content)

                elif chunk.type == "tool_call":
                    # Stop live display before showing tool panel
                    if live is not None:
                        live.stop()
                        live = None
                        text_buffer = ""
                    # Tool call display is handled by _confirm_tool callback

                elif chunk.type == "tool_result":
                    renderer.show_tool_result(chunk.tool_name, chunk.content, chunk.tool_args)

                elif chunk.type == "stats":
                    try:
                        data = json.loads(chunk.content)
                        turn_prompt = data.get("prompt_eval_count", 0)
                        turn_completion = data.get("eval_count", 0)
                        turn_eval_duration = data.get("eval_duration", 0)
                        self.session_stats.add(turn_prompt, turn_completion)
                    except (json.JSONDecodeError, KeyError):
                        pass

                elif chunk.type == "debug":
                    renderer.show_debug(chunk.content)

                elif chunk.type == "error":
                    if live is not None:
                        live.stop()
                        live = None
                        text_buffer = ""
                    renderer.show_error(chunk.content)

                elif chunk.type == "done":
                    break

        except KeyboardInterrupt:
            self.llm.cancel()
            renderer.show_info("\n[cancelled]")
        finally:
            renderer.stop_thinking_spinner(spinner)
            if live is not None:
                live.stop()

        # Show stats after response
        if turn_prompt or turn_completion:
            renderer.show_stats(
                turn_prompt, turn_completion, turn_eval_duration,
                self.session_stats.total_tokens, self.session_stats.estimated_cost,
            )

        console.print()

    def run(self, initial_prompt: str | None = None):
        """Main REPL loop."""
        renderer.show_welcome(self.config.model, self.config.working_dir, self.config.model_info)

        # Handle initial prompt
        if initial_prompt:
            renderer.show_info(f"> {initial_prompt[:200]}{'...' if len(initial_prompt) > 200 else ''}")
            chunks = self.llm.send_message(initial_prompt)
            self._process_response(chunks)

        while self._running:
            try:
                prompt = get_prompt_text(self.config.model)
                user_input = self.prompt_session.prompt(prompt).strip()

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    if self._handle_slash_command(user_input):
                        continue

                # Send to LLM and stream response
                chunks = self.llm.send_message(user_input)
                self._process_response(chunks)

            except KeyboardInterrupt:
                console.print()
                continue
            except EOFError:
                # Ctrl+D
                console.print("\n[dim]Goodbye![/dim]")
                break
            except Exception as e:
                renderer.show_error(f"Unexpected error: {e}")
                continue

        # Auto-save on exit
        try:
            auto_save_session(self.llm.messages)
        except Exception:
            pass

        console.print("[dim]Goodbye![/dim]")

    def run_print(self, prompt: str):
        """Non-interactive print mode: send prompt, output raw text, exit."""
        chunks = self.llm.send_message(prompt)

        try:
            for chunk in chunks:
                if chunk.type == "text":
                    sys.stdout.write(chunk.content)
                    sys.stdout.flush()
                elif chunk.type == "tool_result":
                    # In print mode, show tool results on stderr
                    sys.stderr.write(f"[tool:{chunk.tool_name}] {chunk.content[:500]}\n")
                elif chunk.type == "error":
                    sys.stderr.write(f"Error: {chunk.content}\n")
                elif chunk.type == "done":
                    break
        except KeyboardInterrupt:
            self.llm.cancel()
            sys.stderr.write("\n[cancelled]\n")

        sys.stdout.write("\n")
        sys.stdout.flush()
