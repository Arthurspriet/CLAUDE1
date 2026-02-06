"""Main REPL orchestration - connects input, LLM, and display."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from config import AppConfig, DATA_DIR, parse_model_spec
from doctor import run_health_checks
from model_profiles import get_profile, format_profile_info
from llm import LLMInterface
from session import save_session, load_session, list_sessions, auto_save_session, get_latest_session, export_as_markdown
from skills import SkillRegistry
from stats import SessionStats
from task_manager import TaskManager
from tools import ToolRegistry
from ui.prompts import create_prompt_session, get_prompt_text
from ui import renderer

console = Console()


class REPL:
    """Interactive REPL that orchestrates input -> LLM -> display."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.auto_accept = config.auto_accept_tools
        self.task_manager = TaskManager()
        self.tool_registry = ToolRegistry(
            config.working_dir,
            config=config,
            task_manager=self.task_manager,
            confirm_callback=self._confirm_tool,
        )
        self.llm = LLMInterface(
            config=config,
            tool_registry=self.tool_registry,
            confirm_callback=self._confirm_tool,
        )
        self.skill_registry = SkillRegistry(config.working_dir)
        self.prompt_session = create_prompt_session()
        self._running = True
        self.session_stats = SessionStats()
        self._tool_history: list[str] = []

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
                renderer.show_info(f"Current model: {self.config.model} (provider: {self.config.provider})")
                renderer.show_info("Usage: /model <name>  (prefix with 'hf:' for HuggingFace)")
                return True
            provider, model_id = parse_model_spec(arg)
            self.config.model = arg
            self.config.provider = provider
            # Resolve profile for new model
            ollama_family = None
            if provider == "ollama":
                try:
                    import ollama
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
                import ollama
                models = ollama.list()
                names = [m.model for m in models.models]
                renderer.show_models(sorted(names), self.config.model)
            except Exception as e:
                renderer.show_error(f"Failed to list Ollama models: {e}")
            renderer.show_info("Tip: Use 'hf:<org>/<model>' for HuggingFace models (e.g. hf:meta-llama/Meta-Llama-3-8B-Instruct)")
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

        elif cmd == "/doctor":
            results = run_health_checks(self.config, self.tool_registry)
            renderer.show_doctor_results(results)
            return True

        elif cmd == "/debug":
            data = self._build_debug_info()
            renderer.show_debug_info(data)
            return True

        elif cmd == "/context":
            usage = self.llm.get_context_usage()
            renderer.show_context_usage(usage)
            return True

        elif cmd in ("/stats", "/cost"):
            renderer.show_session_stats(self.session_stats)
            return True

        elif cmd == "/copy":
            self._handle_copy()
            return True

        elif cmd == "/rewind":
            self._handle_rewind(arg)
            return True

        elif cmd == "/memory":
            self._handle_memory(arg)
            return True

        elif cmd == "/init":
            self._handle_init()
            return True

        elif cmd == "/skills":
            self._handle_skills(arg)
            return True

        elif cmd == "/tasks":
            renderer.show_tasks(self.task_manager.list_tasks())
            return True

        elif cmd == "/agents":
            return self._handle_agents_command(arg)

        elif cmd in ("/exit", "/quit"):
            self._running = False
            return True

        # Check if it's a skill invocation
        skill_name = cmd.lstrip("/")
        skill = self.skill_registry.get(skill_name)
        if skill:
            rendered = self.skill_registry.render(skill, arg)
            renderer.show_skill_activated(skill.name, skill.description)
            self.llm.set_tool_filter(skill.allowed_tools)
            try:
                chunks = self.llm.send_message(f"[Skill: {skill.name}]\n\n{rendered}")
                self._process_response(chunks)
            finally:
                self.llm.clear_tool_filter()
            return True

        return False

    def _build_debug_info(self) -> dict:
        """Gather session debug information."""
        # Role breakdown (message counts)
        role_counts: dict[str, int] = {}
        for msg in self.llm.messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "model": self.config.model,
            "working_dir": self.config.working_dir,
            "auto_accept": self.auto_accept,
            "compact": self.config.compact,
            "planning": self.config.planning,
            "temperature": self.config.temperature if self.config.temperature is not None else "default",
            "message_count": len(self.llm.messages),
            "role_breakdown": role_counts,
            "estimated_tokens": self.llm._estimate_tokens(),
            "tool_history": self._tool_history[-10:],
        }

    def _handle_copy(self):
        """Copy last assistant response to clipboard."""
        # Find last assistant message
        last_text = None
        for msg in reversed(self.llm.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    last_text = content
                    break

        if not last_text:
            renderer.show_info("No assistant response to copy.")
            return

        if self._copy_to_clipboard(last_text):
            renderer.show_info(f"Copied {len(last_text)} chars to clipboard.")
        else:
            renderer.show_error("No clipboard tool available (tried wl-copy, xclip, xsel, pbcopy).")

    @staticmethod
    def _copy_to_clipboard(text: str) -> bool:
        """Try to copy text to system clipboard. Returns True on success."""
        for cmd in ["wl-copy", "xclip -selection clipboard", "xsel --clipboard --input", "pbcopy"]:
            try:
                proc = subprocess.run(
                    cmd.split(),
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
                if proc.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return False

    def _handle_rewind(self, arg: str):
        """Handle /rewind command."""
        # Find user turns
        user_turns: list[tuple[int, int, str]] = []  # (turn_num, msg_index, preview)
        turn_num = 0
        for i, msg in enumerate(self.llm.messages):
            if msg.get("role") == "user":
                turn_num += 1
                content = msg.get("content", "")
                preview = content[:80].replace("\n", " ") if isinstance(content, str) else str(content)[:80]
                user_turns.append((turn_num, i, preview))

        if not user_turns:
            renderer.show_info("No user turns to rewind.")
            return

        if not arg:
            # Show available turns
            display = [(t[0], t[2]) for t in reversed(user_turns)]
            renderer.show_rewind_options(display)
            return

        try:
            n = int(arg)
        except ValueError:
            renderer.show_error("Usage: /rewind <turn_number>")
            return

        # Find the turn
        target = None
        for turn_num_val, msg_idx, _ in user_turns:
            if turn_num_val == n:
                target = msg_idx
                break

        if target is None:
            renderer.show_error(f"Turn {n} not found. Valid: 1-{len(user_turns)}")
            return

        # Auto-save before rewinding
        try:
            auto_save_session(self.llm.messages)
        except Exception:
            pass

        # Truncate messages to just before the target user message
        self.llm.messages = self.llm.messages[:target]
        renderer.show_info(f"Rewound to before turn {n}. ({len(self.llm.messages)} messages remaining)")

    def _handle_memory(self, arg: str):
        """Handle /memory command with subcommands."""
        parts = arg.strip().split(maxsplit=1) if arg.strip() else ["show"]
        subcmd = parts[0].lower()
        subarg = parts[1] if len(parts) > 1 else ""

        project_path = Path(self.config.working_dir) / "CLAUDE.md"
        global_path = DATA_DIR / "CLAUDE.md"

        if subcmd == "show":
            # Show both project and global CLAUDE.md
            shown = False
            if project_path.exists():
                try:
                    content = project_path.read_text()
                    renderer.show_info(f"[Project CLAUDE.md] {project_path}")
                    console.print(Markdown(content))
                    shown = True
                except OSError as e:
                    renderer.show_error(f"Cannot read {project_path}: {e}")
            if global_path.exists():
                try:
                    content = global_path.read_text()
                    renderer.show_info(f"[Global CLAUDE.md] {global_path}")
                    console.print(Markdown(content))
                    shown = True
                except OSError as e:
                    renderer.show_error(f"Cannot read {global_path}: {e}")
            if not shown:
                renderer.show_info("No CLAUDE.md found. Use /memory add <text> or /memory edit to create one.")

        elif subcmd == "add":
            if not subarg:
                renderer.show_error("Usage: /memory add <text>")
                return
            # Append to project CLAUDE.md
            try:
                existing = project_path.read_text() if project_path.exists() else ""
                with open(project_path, "a") as f:
                    if existing and not existing.endswith("\n"):
                        f.write("\n")
                    f.write(subarg + "\n")
                renderer.show_info(f"Added to {project_path}")
                self.llm._set_system_prompt()
            except OSError as e:
                renderer.show_error(f"Cannot write: {e}")

        elif subcmd == "edit":
            # Open project CLAUDE.md in $EDITOR
            if not project_path.exists():
                project_path.write_text("# Project Instructions\n\n")
            editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
            try:
                subprocess.run([editor, str(project_path)], check=False)
                self.llm._set_system_prompt()
                renderer.show_info("CLAUDE.md updated. System prompt refreshed.")
            except FileNotFoundError:
                renderer.show_error(f"Editor not found: {editor}")

        elif subcmd == "global":
            # Open global CLAUDE.md in $EDITOR
            if not global_path.exists():
                global_path.write_text("# Global Instructions\n\n")
            editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
            try:
                subprocess.run([editor, str(global_path)], check=False)
                self.llm._set_system_prompt()
                renderer.show_info("Global CLAUDE.md updated. System prompt refreshed.")
            except FileNotFoundError:
                renderer.show_error(f"Editor not found: {editor}")

        elif subcmd == "reset":
            if project_path.exists():
                project_path.unlink()
                renderer.show_info(f"Deleted {project_path}")
                self.llm._set_system_prompt()
            else:
                renderer.show_info("No project CLAUDE.md to reset.")

        else:
            renderer.show_error(f"Unknown subcommand: {subcmd}")
            renderer.show_info("Usage: /memory [show|add <text>|edit|global|reset]")

    def _handle_init(self):
        """Initialize project CLAUDE.md by auto-detecting project info."""
        project_path = Path(self.config.working_dir) / "CLAUDE.md"

        if project_path.exists():
            renderer.show_info(f"CLAUDE.md already exists at {project_path}")
            renderer.show_info("Use /memory edit to modify it.")
            return

        lines = ["# Project Instructions\n"]

        working_dir = Path(self.config.working_dir)

        # Detect languages/frameworks
        detections = []
        if list(working_dir.glob("*.py")) or (working_dir / "pyproject.toml").exists():
            detections.append("Python")
        if (working_dir / "package.json").exists():
            detections.append("Node.js")
        if list(working_dir.glob("*.go")):
            detections.append("Go")
        if list(working_dir.glob("*.rs")) or (working_dir / "Cargo.toml").exists():
            detections.append("Rust")
        if list(working_dir.glob("*.java")) or (working_dir / "pom.xml").exists():
            detections.append("Java")
        if (working_dir / "Makefile").exists():
            detections.append("Make")
        if (working_dir / "Dockerfile").exists():
            detections.append("Docker")

        if detections:
            lines.append(f"## Stack\n{', '.join(detections)}\n")

        # Detect test framework
        test_info = []
        if (working_dir / "pytest.ini").exists() or (working_dir / "pyproject.toml").exists():
            test_info.append("pytest (run: `pytest`)")
        if (working_dir / "jest.config.js").exists() or (working_dir / "jest.config.ts").exists():
            test_info.append("Jest (run: `npm test`)")
        if test_info:
            lines.append(f"## Tests\n{', '.join(test_info)}\n")

        # Git info
        try:
            subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           cwd=self.config.working_dir, capture_output=True, check=True, timeout=5)
            lines.append("## Version Control\nThis project uses git.\n")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        lines.append("## Conventions\n- Follow existing code style\n- Write clear commit messages\n")

        project_path.write_text("\n".join(lines))
        renderer.show_info(f"Created {project_path}")
        self.llm._set_system_prompt()
        renderer.show_info("System prompt updated with project instructions.")

    def _handle_skills(self, arg: str):
        """Handle /skills command with subcommands."""
        parts = arg.strip().split(maxsplit=1) if arg.strip() else ["list"]
        subcmd = parts[0].lower()
        subarg = parts[1].strip() if len(parts) > 1 else ""

        if subcmd == "list" or subcmd == "":
            renderer.show_skills(self.skill_registry.list_skills())

        elif subcmd == "create":
            if not subarg:
                renderer.show_error("Usage: /skills create <name>")
                return
            skill_file = self.skill_registry.create_skill(subarg)
            if skill_file is None:
                renderer.show_error(f"Skill '{subarg}' already exists.")
                return
            renderer.show_skill_created(subarg, skill_file)
            # Open in editor
            self.skill_registry.open_in_editor(skill_file)
            # Reload after editing
            self.skill_registry.reload()
            skill = self.skill_registry.get(subarg)
            if skill:
                renderer.show_info(f"Skill '{subarg}' registered. Use /{subarg} to invoke it.")
            else:
                renderer.show_info("Skill file saved but could not be parsed. Check the YAML frontmatter.")

        elif subcmd == "edit":
            if not subarg:
                renderer.show_error("Usage: /skills edit <name>")
                return
            skill = self.skill_registry.get(subarg)
            if not skill:
                renderer.show_error(f"Skill '{subarg}' not found.")
                return
            skill_file = skill.path / "SKILL.md"
            self.skill_registry.open_in_editor(skill_file)
            self.skill_registry.reload()
            renderer.show_info(f"Skill '{subarg}' reloaded.")

        elif subcmd == "reload":
            self.skill_registry.reload()
            count = len(self.skill_registry.list_skills())
            renderer.show_info(f"Reloaded. {count} skill(s) found.")

        elif subcmd == "info":
            if not subarg:
                renderer.show_error("Usage: /skills info <name>")
                return
            skill = self.skill_registry.get(subarg)
            if not skill:
                renderer.show_error(f"Skill '{subarg}' not found.")
                return
            renderer.show_skill_info(skill)

        else:
            renderer.show_error(f"Unknown subcommand: {subcmd}")
            renderer.show_info("Usage: /skills [list|create <name>|edit <name>|reload|info <name>]")

    def _handle_agents_command(self, arg: str) -> bool:
        """Handle /agents command and subcommands."""
        from agents.config import DEFAULT_AGENT_CONFIGS, AgentRoleConfig

        parts = arg.strip().split(maxsplit=2) if arg.strip() else []

        if not parts:
            # Toggle agents mode
            self.config.agents_mode = not self.config.agents_mode
            state = "ON" if self.config.agents_mode else "OFF"
            renderer.show_info(f"Multi-agent mode: {state}")
            return True

        subcmd = parts[0].lower()

        if subcmd == "config":
            renderer.show_agent_config(DEFAULT_AGENT_CONFIGS)
            return True

        elif subcmd == "set":
            if len(parts) < 3:
                renderer.show_error("Usage: /agents set <role> <model>")
                renderer.show_info(f"Available roles: {', '.join(DEFAULT_AGENT_CONFIGS.keys())}")
                return True
            role_name = parts[1].lower()
            model_name = parts[2]
            if role_name not in DEFAULT_AGENT_CONFIGS:
                renderer.show_error(f"Unknown role: {role_name}")
                renderer.show_info(f"Available roles: {', '.join(DEFAULT_AGENT_CONFIGS.keys())}")
                return True
            old_config = DEFAULT_AGENT_CONFIGS[role_name]
            DEFAULT_AGENT_CONFIGS[role_name] = AgentRoleConfig(
                model=model_name,
                tool_names=old_config.tool_names,
                read_only=old_config.read_only,
                max_iterations=old_config.max_iterations,
                system_prompt_extra=old_config.system_prompt_extra,
            )
            renderer.show_info(f"Agent role '{role_name}' now uses model: {model_name}")
            return True

        else:
            renderer.show_error(f"Unknown subcommand: {subcmd}")
            renderer.show_info("Usage: /agents [config|set <role> <model>]")
            return True

    def _run_agents(self, user_input: str):
        """Run user input through the multi-agent orchestrator."""
        from agents.orchestrator import AgentOrchestrator, OrchestratorEventType
        from agents.config import DEFAULT_AGENT_CONFIGS
        from session import add_agent_run_message

        orchestrator = AgentOrchestrator(
            tool_registry=self.tool_registry,
            working_dir=self.config.working_dir,
            ollama_host=self.config.ollama_host,
            config_overrides=DEFAULT_AGENT_CONFIGS,
        )

        async def _run():
            return await orchestrator.run(user_input)

        # Run with Ctrl+C support
        try:
            spinner = renderer.start_thinking_spinner()
            try:
                final_output, events = asyncio.run(_run())
            finally:
                renderer.stop_thinking_spinner(spinner)

            # Render events
            for event in events:
                if event.type == OrchestratorEventType.PLAN and event.plan:
                    renderer.show_agent_plan(event.plan.summary, event.plan.tasks)
                elif event.type == OrchestratorEventType.AGENT_START:
                    renderer.show_agent_progress(
                        event.task_id, event.role, event.model, "running",
                    )
                elif event.type == OrchestratorEventType.AGENT_DONE:
                    renderer.show_agent_progress(
                        event.task_id, event.role, event.model, event.status,
                    )
                    # Track stats
                    if event.result and event.result.token_usage:
                        usage = event.result.token_usage
                        self.session_stats.add_agent_run(
                            role=event.role,
                            model=event.model,
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                        )
                    # Store in session
                    if event.result:
                        add_agent_run_message(
                            self.llm.messages,
                            role=event.role,
                            model=event.model,
                            task_id=event.task_id,
                            output=event.result.output,
                            files_modified=event.result.files_modified,
                        )

            # Render final output
            renderer.render_agent_result(final_output)

        except KeyboardInterrupt:
            orchestrator.cancel()
            renderer.show_info("\n[agents cancelled]")

        console.print()

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
                    # Track tool call in history
                    self._tool_history.append(f"{chunk.tool_name}({', '.join(f'{k}={str(v)[:30]}' for k, v in chunk.tool_args.items())})")
                    # Tool call display is handled by _confirm_tool callback

                elif chunk.type == "tool_result":
                    renderer.show_tool_result(chunk.tool_name, chunk.content, chunk.tool_args)

                elif chunk.type == "stats":
                    try:
                        data = json.loads(chunk.content)
                        turn_prompt = data.get("prompt_eval_count", 0)
                        turn_completion = data.get("eval_count", 0)
                        turn_eval_duration = data.get("eval_duration", 0)
                        self.session_stats.add(turn_prompt, turn_completion, turn_eval_duration)
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
            if self.config.agents_mode:
                self._run_agents(initial_prompt)
            else:
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

                # Dispatch: agents mode or single-agent mode
                if self.config.agents_mode:
                    self._run_agents(user_input)
                else:
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
