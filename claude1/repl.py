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

from claude1.config import AppConfig, DATA_DIR, parse_model_spec
from claude1.doctor import run_health_checks
from claude1.model_profiles import get_profile, format_profile_info
from claude1.llm import LLMInterface
from claude1.session import (
    save_session, load_session, list_sessions, auto_save_session,
    get_latest_session, export_as_markdown,
    save_checkpoint, load_checkpoint, list_checkpoints, delete_checkpoint,
)
from claude1.skills import SkillRegistry
from claude1.stats import SessionStats
from claude1.task_engine import TaskEngine, TaskTimeManager
from claude1.tools import ToolRegistry
from claude1.ui.prompts import create_prompt_session, get_prompt_text
from claude1.ui import renderer

console = Console()


class REPL:
    """Interactive REPL that orchestrates input -> LLM -> display."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.auto_accept = config.auto_accept_tools
        self.task_engine = TaskEngine()

        # ── MCP: connect to external tool servers ──
        self._mcp_manager = None
        mcp_tools = None
        try:
            from claude1.mcp_client import MCPClientManager
            self._mcp_manager = MCPClientManager(config.working_dir)
            self._mcp_manager.load_config()
            mcp_tools = self._mcp_manager.connect_all()
            if mcp_tools:
                renderer.show_info(f"MCP: {len(mcp_tools)} tool(s) loaded from {len(self._mcp_manager._configs)} server(s)")
        except Exception:
            pass  # MCP is optional

        self.tool_registry = ToolRegistry(
            config.working_dir,
            config=config,
            task_manager=self.task_engine,
            confirm_callback=self._confirm_tool,
            mcp_tools=mcp_tools or None,
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
        self._time_manager = TaskTimeManager()
        self._execution_logger = None
        try:
            from claude1.task_engine.logger import ExecutionLogger
            self._execution_logger = ExecutionLogger()
        except Exception:
            pass

    def _confirm_tool(self, tool_name: str, tool_args: dict) -> bool:
        """Confirmation callback invoked by the LLM before executing dangerous tools."""
        # For bash commands, check safety classification and force confirmation for warned commands
        if tool_name == "bash":
            from claude1.tools.bash_tool import CommandSafety
            command = tool_args.get("command", "")
            level, _reason = CommandSafety.classify(command)
            if level == CommandSafety.WARNED:
                renderer.show_info(f"[WARNING] Potentially dangerous command detected.")
                approved, always = renderer.confirm_tool(tool_name, tool_args)
                if always:
                    self.auto_accept = True
                    renderer.show_info("Auto-accept enabled for this session.")
                return approved

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

        elif cmd == "/undo-bash":
            if arg.strip().lower() == "list":
                snapshots = self.tool_registry.bash_undo.list_snapshots()
                if not snapshots:
                    renderer.show_info("No bash snapshots available.")
                else:
                    renderer.show_info(f"Bash undo snapshots ({len(snapshots)}):")
                    for s in snapshots:
                        renderer.show_info(f"  [{s['ref']}] {s['command']}")
            else:
                result = self.tool_registry.bash_undo.undo_last()
                renderer.show_info(result)
            return True

        elif cmd == "/checkpoint":
            if not arg:
                # List checkpoints
                cps = list_checkpoints()
                if not cps:
                    renderer.show_info("No checkpoints saved. Usage: /checkpoint <name>")
                else:
                    renderer.show_info(f"Checkpoints ({len(cps)}):")
                    for cp in cps:
                        renderer.show_info(f"  {cp['name']}  ({cp['messages']} msgs, {cp['checkpoint_at']})")
                return True

            subcmd_parts = arg.strip().split(maxsplit=1)
            subcmd_name = subcmd_parts[0].lower()

            if subcmd_name == "delete" and len(subcmd_parts) > 1:
                cp_name = subcmd_parts[1].strip()
                if delete_checkpoint(cp_name):
                    renderer.show_info(f"Checkpoint '{cp_name}' deleted.")
                else:
                    renderer.show_error(f"Checkpoint '{cp_name}' not found.")
                return True

            # Save checkpoint
            path = save_checkpoint(self.llm.messages, arg.strip())
            renderer.show_info(f"Checkpoint saved: {arg.strip()} ({len(self.llm.messages)} messages)")
            return True

        elif cmd == "/branch":
            if not arg:
                renderer.show_info("Usage: /branch <checkpoint_name>")
                renderer.show_info("Forks conversation from a checkpoint (non-destructive).")
                cps = list_checkpoints()
                if cps:
                    renderer.show_info(f"\nAvailable checkpoints:")
                    for cp in cps:
                        renderer.show_info(f"  {cp['name']}  ({cp['messages']} msgs)")
                return True

            # Auto-save current state before branching
            try:
                auto_save_session(self.llm.messages)
            except Exception:
                pass

            messages = load_checkpoint(arg.strip())
            if messages is None:
                renderer.show_error(f"Checkpoint '{arg.strip()}' not found.")
            else:
                self.llm.messages = messages
                renderer.show_info(
                    f"Branched from checkpoint '{arg.strip()}' ({len(messages)} messages). "
                    f"Previous state auto-saved."
                )
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
            if arg.strip().lower() == "history":
                from claude1.stats import load_persistent_stats
                history = load_persistent_stats()
                renderer.show_persistent_stats(history)
            else:
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
            renderer.show_tasks(self.task_engine.list_tasks())
            return True

        elif cmd == "/hf-search":
            self._handle_hf_search(arg)
            return True

        elif cmd == "/hf-import":
            self._handle_hf_import(arg)
            return True

        elif cmd == "/agents":
            return self._handle_agents_command(arg)

        elif cmd == "/video-text":
            self._handle_video_text(arg)
            return True

        elif cmd == "/video-image":
            self._handle_video_image(arg)
            return True

        elif cmd == "/video-models":
            self._handle_video_models(arg)
            return True

        elif cmd == "/verify":
            self._handle_verify(arg)
            return True

        elif cmd == "/bandit":
            self._handle_bandit(arg)
            return True

        elif cmd == "/mission":
            self._handle_mission(arg)
            return True

        elif cmd == "/resume-task":
            self._handle_resume_task(arg)
            return True

        elif cmd == "/capabilities":
            self._handle_capabilities()
            return True

        elif cmd == "/create-skill":
            self._handle_create_skill(arg)
            return True

        elif cmd == "/create-role":
            self._handle_create_role(arg)
            return True

        elif cmd == "/self-modifications":
            self._handle_self_modifications()
            return True

        elif cmd == "/revert":
            self._handle_revert()
            return True

        elif cmd == "/reflect":
            self._handle_reflect()
            return True

        elif cmd == "/analytics":
            self._handle_analytics()
            return True

        elif cmd == "/time-budget":
            self._handle_time_budget(arg)
            return True

        elif cmd == "/mcp":
            self._handle_mcp(arg)
            return True

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

    def _handle_hf_search(self, arg: str):
        """Handle /hf-search command -- search HuggingFace for GGUF models."""
        if not arg.strip():
            renderer.show_info("Usage: /hf-search <query>")
            renderer.show_info("Example: /hf-search dolphin uncensored")
            return

        try:
            from claude1.hf_import import search_hf_gguf
        except ImportError as e:
            renderer.show_error(f"HF import module unavailable: {e}")
            return

        renderer.show_info(f"Searching HuggingFace for: {arg.strip()}...")
        try:
            results = search_hf_gguf(arg.strip())
            renderer.show_hf_search_results(results)
        except Exception as e:
            renderer.show_error(f"Search failed: {e}")

    def _handle_hf_import(self, arg: str):
        """Handle /hf-import command -- download GGUF from HF and import into Ollama."""
        if not arg.strip():
            renderer.show_info("Usage: /hf-import <repo_id> [model_name]")
            renderer.show_info("Example: /hf-import bartowski/Dolphin3.0-Llama3.1-8B-GGUF")
            renderer.show_info("Use /hf-search <query> to find models first.")
            return

        try:
            from claude1.hf_import import full_import, list_gguf_files, get_memory_budget, ImportProgress
        except ImportError as e:
            renderer.show_error(f"HF import module unavailable: {e}")
            return

        parts = arg.strip().split(maxsplit=1)
        repo_id = parts[0]
        custom_name = parts[1] if len(parts) > 1 else None

        # Show what we're about to do
        renderer.show_info(f"Importing from: {repo_id}")
        if custom_name:
            renderer.show_info(f"Custom model name: {custom_name}")

        def on_progress(progress: ImportProgress):
            renderer.show_import_progress(progress.stage, progress.message)

        try:
            model_name = full_import(
                repo_id=repo_id,
                model_name=custom_name,
                num_ctx=self.config.num_ctx,
                progress_callback=on_progress,
            )
            renderer.show_import_complete(model_name)
        except KeyboardInterrupt:
            renderer.show_info("\n[import cancelled]")
        except Exception as e:
            renderer.show_error(f"Import failed: {e}")

    def _handle_video_text(self, arg: str):
        """Handle /video-text command -- generate video from text prompt."""
        import shlex

        if not arg.strip():
            renderer.show_info("Usage: /video-text <prompt> [--model MODEL] [--frames N] [--width W] [--height H] [--fps F]")
            renderer.show_info("Example: /video-text 'a cat playing piano' --frames 30")
            return

        try:
            from claude1.video_generation import VideoGenerator
            from claude1.video_models import get_default_model
        except ImportError as e:
            renderer.show_error(f"Video generation dependencies not installed: {e}")
            renderer.show_info("Install with: pip install diffusers torch torchvision transformers accelerate imageio[ffmpeg]")
            return

        # Parse arguments (simple parser for --key value pairs)
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            renderer.show_error("Prompt is required")
            return

        prompt = ""
        model_id = None
        num_frames = None
        width = None
        height = None
        fps = None
        output_filename = None

        i = 0
        while i < len(parts):
            if parts[i] == "--model" and i + 1 < len(parts):
                model_id = parts[i + 1]
                i += 2
            elif parts[i] == "--frames" and i + 1 < len(parts):
                try:
                    num_frames = int(parts[i + 1])
                except ValueError:
                    renderer.show_error(f"Invalid frames value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--width" and i + 1 < len(parts):
                try:
                    width = int(parts[i + 1])
                except ValueError:
                    renderer.show_error(f"Invalid width value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--height" and i + 1 < len(parts):
                try:
                    height = int(parts[i + 1])
                except ValueError:
                    renderer.show_error(f"Invalid height value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--fps" and i + 1 < len(parts):
                try:
                    fps = int(parts[i + 1])
                except ValueError:
                    renderer.show_error(f"Invalid fps value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--output" and i + 1 < len(parts):
                output_filename = parts[i + 1]
                i += 2
            else:
                # Collect as prompt text
                if prompt:
                    prompt += " " + parts[i]
                else:
                    prompt = parts[i]
                i += 1

        if not prompt:
            renderer.show_error("Prompt is required")
            return

        if model_id is None:
            model_id = get_default_model("text-to-video")

        renderer.show_info(f"Generating video from text: '{prompt}'")
        renderer.show_info(f"Model: {model_id}")

        def on_progress(pct: float):
            renderer.show_info(f"Progress: {pct*100:.1f}%")

        try:
            generator = VideoGenerator()
            output_path = generator.generate_text_to_video(
                prompt=prompt,
                model_id=model_id,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                output_filename=output_filename,
                progress_callback=on_progress,
            )
            renderer.show_info(f"Video generated: {output_path}")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            renderer.show_info(f"File size: {file_size_mb:.2f} MB")
        except Exception as e:
            renderer.show_error(f"Video generation failed: {e}")

    def _handle_video_image(self, arg: str):
        """Handle /video-image command -- generate video from image."""
        import shlex

        if not arg.strip():
            renderer.show_info("Usage: /video-image <image_path> [--prompt PROMPT] [--model MODEL] [--frames N] [--width W] [--height H] [--fps F]")
            renderer.show_info("Example: /video-image photo.jpg --prompt 'make it move'")
            return

        try:
            from claude1.video_generation import VideoGenerator
            from claude1.video_models import get_default_model
        except ImportError as e:
            renderer.show_error(f"Video generation dependencies not installed: {e}")
            renderer.show_info("Install with: pip install diffusers torch torchvision transformers accelerate imageio[ffmpeg]")
            return

        # Parse arguments
        parts = shlex.split(arg) if arg.strip() else []
        if not parts:
            renderer.show_error("Image path is required")
            return

        image_path = None
        prompt = None
        model_id = None
        num_frames = None
        width = None
        height = None
        fps = None
        output_filename = None

        i = 0
        while i < len(parts):
            if parts[i] == "--prompt" and i + 1 < len(parts):
                prompt = parts[i + 1]
                i += 2
            elif parts[i] == "--model" and i + 1 < len(parts):
                model_id = parts[i + 1]
                i += 2
            elif parts[i] == "--frames" and i + 1 < len(parts):
                try:
                    num_frames = int(parts[i + 1])
                except ValueError:
                    renderer.show_error(f"Invalid frames value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--width" and i + 1 < len(parts):
                try:
                    width = int(parts[i + 1])
                except ValueError:
                    renderer.show_error(f"Invalid width value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--height" and i + 1 < len(parts):
                try:
                    height = int(parts[i + 1])
                except ValueError:
                    renderer.show_error(f"Invalid height value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--fps" and i + 1 < len(parts):
                try:
                    fps = int(parts[i + 1])
                except ValueError:
                    renderer.show_error(f"Invalid fps value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--output" and i + 1 < len(parts):
                output_filename = parts[i + 1]
                i += 2
            else:
                # First non-flag argument is the image path
                if image_path is None:
                    image_path = parts[i]
                i += 1

        if not image_path:
            renderer.show_error("Image path is required")
            return

        # Resolve image path
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = Path(self.config.working_dir) / image_path

        if not image_path.exists():
            renderer.show_error(f"Image not found: {image_path}")
            return

        if model_id is None:
            model_id = get_default_model("image-to-video")

        renderer.show_info(f"Generating video from image: {image_path}")
        renderer.show_info(f"Model: {model_id}")
        if prompt:
            renderer.show_info(f"Prompt: {prompt}")

        def on_progress(pct: float):
            renderer.show_info(f"Progress: {pct*100:.1f}%")

        try:
            generator = VideoGenerator()
            output_path = generator.generate_image_to_video(
                image_path=image_path,
                prompt=prompt,
                model_id=model_id,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                output_filename=output_filename,
                progress_callback=on_progress,
            )
            renderer.show_info(f"Video generated: {output_path}")
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            renderer.show_info(f"File size: {file_size_mb:.2f} MB")
        except Exception as e:
            renderer.show_error(f"Video generation failed: {e}")

    def _handle_video_models(self, arg: str):
        """Handle /video-models command -- list available video generation models."""
        try:
            from claude1.video_models import list_models, VIDEO_MODELS
        except ImportError as e:
            renderer.show_error(f"Video generation dependencies not installed: {e}")
            renderer.show_info("Install with: pip install diffusers torch torchvision transformers accelerate imageio[ffmpeg]")
            return

        capability = None
        if arg.strip():
            capability = arg.strip().lower()
            if capability not in ["text-to-video", "image-to-video"]:
                renderer.show_error(f"Unknown capability: {capability}")
                renderer.show_info("Usage: /video-models [text-to-video|image-to-video]")
                return

        models = list_models(capability) if capability else list(VIDEO_MODELS.values())

        if not models:
            renderer.show_info("No models found.")
            return

        from rich.table import Table

        table = Table(title="Available Video Generation Models", border_style="dim")
        table.add_column("Model ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Capabilities", style="yellow")
        table.add_column("Memory (GB)", style="magenta")
        table.add_column("Recommended", style="blue")

        for model in models:
            caps = ", ".join(model.capabilities)
            rec = f"{model.recommended_frames}f @ {model.recommended_width}x{model.recommended_height} {model.recommended_fps}fps"
            table.add_row(
                model.model_id,
                model.name,
                caps,
                f"{model.memory_gb:.1f}",
                rec,
            )

        console.print(table)

    def _handle_verify(self, arg: str):
        """Handle /verify command -- critique and refine last response."""
        from claude1.critique import run_self_critique

        subcmd = arg.strip().lower() if arg.strip() else ""

        if subcmd == "auto":
            self.config.auto_verify = not self.config.auto_verify
            state = "ON" if self.config.auto_verify else "OFF"
            renderer.show_info(f"Auto-verify: {state}")
            if self.config.auto_verify:
                renderer.show_info("Responses will be automatically critiqued and refined (adds latency).")
            return

        # Find last assistant response
        last_text = None
        context = ""
        for msg in reversed(self.llm.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    last_text = content
                    break
            elif msg.get("role") == "user" and not context:
                context = msg.get("content", "")

        if not last_text:
            renderer.show_info("No assistant response to verify.")
            return

        renderer.show_info("Running self-critique...")
        spinner = renderer.start_thinking_spinner()
        try:
            result = run_self_critique(self.llm, last_text, context)
        finally:
            renderer.stop_thinking_spinner(spinner)

        if not result.success:
            renderer.show_error(f"Critique failed: {result.error}")
            return

        # Display critique
        renderer.show_info("--- Critique ---")
        console.print(Markdown(result.critique))

        # Display refined output
        renderer.show_info(f"--- Refined Output (improvement: {result.improvement_score:.0%}) ---")
        console.print(Markdown(result.refined_output))

        # Replace the last assistant message with the refined version
        for i in range(len(self.llm.messages) - 1, -1, -1):
            if self.llm.messages[i].get("role") == "assistant":
                self.llm.messages[i]["content"] = result.refined_output
                break

        renderer.show_info("Last response replaced with refined version.")

    def _handle_mcp(self, arg: str):
        """Handle /mcp command with subcommands."""
        parts = arg.strip().split(maxsplit=1) if arg.strip() else ["status"]
        subcmd = parts[0].lower()
        subarg = parts[1].strip() if len(parts) > 1 else ""

        if subcmd in ("status", ""):
            if self._mcp_manager is None:
                renderer.show_info("MCP: not initialized (mcp package not installed?)")
                renderer.show_info("Install with: pip install 'claude1[mcp]'")
                return
            status = self._mcp_manager.get_status()
            renderer.show_info(f"MCP available: {status['mcp_available']}")
            renderer.show_info(f"Total MCP tools: {status['total_tools']}")
            if status["servers"]:
                from rich.table import Table
                table = Table(title="MCP Servers", border_style="dim")
                table.add_column("Server", style="cyan")
                table.add_column("Transport", style="green")
                table.add_column("Connected", style="yellow")
                table.add_column("Tools", style="magenta")
                table.add_column("Endpoint", style="blue")
                for name, info in status["servers"].items():
                    endpoint = info.get("url") or info.get("command") or ""
                    table.add_row(
                        name,
                        info["transport"],
                        "yes" if info["connected"] else "no",
                        str(info["tools"]),
                        endpoint,
                    )
                console.print(table)
            else:
                renderer.show_info("No MCP servers configured.")
                renderer.show_info("Add servers to ~/.claude1/mcp.json or .claude1/mcp.json")

        elif subcmd == "reconnect":
            if self._mcp_manager is None:
                renderer.show_error("MCP not initialized.")
                return
            renderer.show_info("Reconnecting MCP servers...")
            new_tools = self._mcp_manager.reconnect_all()
            # Update tool registry with new MCP tools
            # Remove old MCP tools
            self.tool_registry._tools = {
                k: v for k, v in self.tool_registry._tools.items()
                if not k.startswith("mcp_")
            }
            for tool in new_tools:
                self.tool_registry._tools[tool.name] = tool
            self.llm._set_system_prompt()
            renderer.show_info(f"Reconnected: {len(new_tools)} MCP tool(s) available")

        elif subcmd == "add":
            if not subarg:
                # List available templates
                from claude1.mcp_templates import list_templates
                templates = list_templates()
                renderer.show_info("Available MCP templates:")
                for t in templates:
                    env_str = ", ".join(t["required_env"])
                    renderer.show_info(f"  {t['name']}  (requires: {env_str})")
                renderer.show_info("\nUsage: /mcp add <template_name>")
                return

            from claude1.mcp_templates import get_template
            template = get_template(subarg)
            if template is None:
                renderer.show_error(f"Unknown template: {subarg}")
                renderer.show_info("Use /mcp add to see available templates.")
                return

            if self._mcp_manager is None:
                renderer.show_error("MCP not initialized.")
                return

            self._mcp_manager.save_config(subarg, template, global_scope=True)
            renderer.show_info(f"Template '{subarg}' added to ~/.claude1/mcp.json")
            renderer.show_info("Edit the config to set required environment variables, then /mcp reconnect")

        elif subcmd == "config":
            # Open mcp.json in $EDITOR
            config_path = self._mcp_manager.GLOBAL_CONFIG if self._mcp_manager else DATA_DIR / "mcp.json"
            if not config_path.exists():
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text('{\n  "mcpServers": {}\n}\n')
            editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
            try:
                subprocess.run([editor, str(config_path)], check=False)
                renderer.show_info(f"MCP config updated: {config_path}")
                renderer.show_info("Run /mcp reconnect to apply changes.")
            except FileNotFoundError:
                renderer.show_error(f"Editor not found: {editor}")

        else:
            renderer.show_error(f"Unknown subcommand: {subcmd}")
            renderer.show_info("Usage: /mcp [status|reconnect|add <template>|config]")

    def _handle_bandit(self, arg: str):
        """Handle /bandit command -- multi-armed bandit model selection."""
        from claude1.bandit import get_model_bandit

        subcmd = arg.strip().lower() if arg.strip() else ""

        bandit = get_model_bandit()

        if subcmd == "on":
            self.config.bandit_enabled = True
            renderer.show_info("Bandit model selection: ON")
            renderer.show_info("The bandit will learn which model works best and auto-select over time.")
            return

        if subcmd == "off":
            self.config.bandit_enabled = False
            renderer.show_info("Bandit model selection: OFF")
            return

        if subcmd == "auto":
            if not self.config.bandit_enabled:
                renderer.show_info("Bandit is not enabled. Use /bandit on first.")
                return
            selected = bandit.select()
            if selected and selected != self.config.model:
                # Switch model
                from claude1.config import parse_model_spec
                provider, model_id = parse_model_spec(selected)
                self.config.model = selected
                self.config.provider = provider
                profile = get_profile(selected)
                self.config.profile = profile
                self.config.num_ctx = profile.num_ctx
                self.llm.update_model(selected)
                renderer.show_model_changed(selected)
                renderer.show_info(f"Bandit selected: {selected}")
            else:
                renderer.show_info(f"Bandit recommends current model: {self.config.model}")
            return

        if subcmd == "reset":
            bandit.reset()
            renderer.show_info("Bandit state reset. All learned preferences cleared.")
            return

        # Default: show stats
        state = "ON" if self.config.bandit_enabled else "OFF"
        renderer.show_info(f"Bandit model selection: {state}")

        stats = bandit.get_stats()
        if not stats.get("initialized") or not stats.get("arms"):
            renderer.show_info("No bandit data yet. Enable with /bandit on and use different models.")
            return

        renderer.show_info(f"Total pulls: {stats.get('total_pulls', 0)}")
        renderer.show_info(f"Exploration complete: {'yes' if stats.get('exploration_complete') else 'no'}")
        if stats.get("best_arm"):
            renderer.show_info(f"Best model: {stats['best_arm']}")

        from rich.table import Table
        table = Table(title="Bandit Arms (Models)", border_style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Pulls", style="green")
        table.add_column("Mean Reward", style="yellow")
        table.add_column("UCB Score", style="magenta")

        for name, arm_stats in stats.get("arms", {}).items():
            table.add_row(
                name,
                str(arm_stats.get("pulls", 0)),
                f"{arm_stats.get('mean_reward', 0):.3f}",
                f"{arm_stats.get('ucb_score', 0):.3f}",
            )

        console.print(table)
        renderer.show_info("Commands: /bandit [on|off|auto|reset]")

    def _handle_mission(self, arg: str):
        """Handle /mission command — start a long multi-agent task with optional time budget."""
        if not arg.strip():
            renderer.show_info("Usage: /mission <description> [--budget Nm]")
            renderer.show_info("Example: /mission refactor auth module --budget 5m")
            return

        import re
        # Parse --budget flag
        budget_match = re.search(r"--budget\s+(\d+)m", arg)
        if budget_match:
            minutes = int(budget_match.group(1))
            self._time_manager.total_budget_seconds = minutes * 60
            arg = re.sub(r"\s*--budget\s+\d+m", "", arg).strip()
            renderer.show_info(f"Time budget: {minutes} minutes")

        # Enable agents mode and run
        was_agents = self.config.agents_mode
        self.config.agents_mode = True
        try:
            self._run_agents(arg)
        finally:
            self.config.agents_mode = was_agents
            self._time_manager.total_budget_seconds = 0

    def _handle_resume_task(self, arg: str):
        """Handle /resume-task command — resume an interrupted execution."""
        if not arg.strip():
            # List available executions
            executions = self.task_engine.list_saved_executions()
            if not executions:
                renderer.show_info("No saved executions found.")
                return
            renderer.show_info("Saved executions:")
            for ex in executions[:10]:
                renderer.show_info(
                    f"  {ex['id']}  {ex['status']}  {ex['description'][:60]}  "
                    f"({ex['task_count']} tasks, {ex['phase_count']} phases)"
                )
            renderer.show_info("Usage: /resume-task <id>")
            return

        plan_id = arg.strip()
        if self.task_engine.resume(plan_id):
            tasks = self.task_engine.list_tasks()
            renderer.show_info(f"Resumed execution {plan_id} ({len(tasks)} tasks)")
            renderer.show_tasks(tasks)
        else:
            renderer.show_error(f"Execution '{plan_id}' not found.")

    def _handle_capabilities(self):
        """Handle /capabilities command — show all tools, skills, agent roles, commands."""
        try:
            from claude1.self_awareness.introspection import CodebaseIntrospector
            introspector = CodebaseIntrospector()
            caps = introspector.get_capabilities()

            # Group by type
            tools = [c for c in caps if c.type == "tool"]
            skills = [c for c in caps if c.type == "skill"]
            roles = [c for c in caps if c.type == "agent_role"]
            commands = [c for c in caps if c.type == "slash_command"]

            renderer.show_info(f"Tools ({len(tools)}):")
            for t in tools:
                renderer.show_info(f"  {t.name}: {t.description}")

            renderer.show_info(f"\nSkills ({len(skills)}):")
            for s in skills:
                renderer.show_info(f"  /{s.name}: {s.description}")

            renderer.show_info(f"\nAgent Roles ({len(roles)}):")
            for r in roles:
                renderer.show_info(f"  {r.name}: {r.description}")

            renderer.show_info(f"\nSlash Commands ({len(commands)}):")
            for c in commands:
                renderer.show_info(f"  {c.name}")
        except Exception as e:
            renderer.show_error(f"Failed to enumerate capabilities: {e}")

    def _handle_create_skill(self, arg: str):
        """Handle /create-skill — interactive model-driven skill creation."""
        if not arg.strip():
            # Send to LLM for interactive creation
            prompt = (
                "The user wants to create a new skill. Ask them what the skill should do, "
                "then use the create_skill tool to create it. A skill is a reusable slash command "
                "that provides specific instructions to the model."
            )
            chunks = self.llm.send_message(prompt)
            self._process_response(chunks)
            return

        # Direct creation: /create-skill <name>
        try:
            from claude1.self_awareness.skill_factory import SkillFactory
            factory = SkillFactory(self.config.working_dir)
            path = factory.create_skill_from_spec(
                name=arg.strip(),
                description="New skill (edit to customize)",
                instructions="## Instructions\n\nDescribe what this skill should do.",
            )
            renderer.show_info(f"Skill '{arg.strip()}' created at {path}")
            renderer.show_info(f"Edit with: /skills edit {arg.strip()}")
            self.skill_registry.reload()
        except Exception as e:
            renderer.show_error(f"Error: {e}")

    def _handle_create_role(self, arg: str):
        """Handle /create-role — interactive model-driven agent role creation."""
        if not arg.strip():
            prompt = (
                "The user wants to create a new agent role. Ask them for: "
                "1) Role name, 2) Which model to use, 3) What tools it should have access to, "
                "4) Any special instructions. Then use the create_agent_role tool."
            )
            chunks = self.llm.send_message(prompt)
            self._process_response(chunks)
            return

        renderer.show_info("Usage: /create-role (no args — interactive mode)")

    def _handle_self_modifications(self):
        """Handle /self-modifications command — show modification history."""
        try:
            from claude1.self_awareness.self_modifier import SelfModifier
            modifier = SelfModifier()
            mods = modifier.list_modifications()
            if not mods:
                renderer.show_info("No self-modifications recorded.")
                return

            import time as time_mod
            renderer.show_info(f"Self-modification history ({len(mods)} total):")
            for m in mods[-15:]:
                ts = time_mod.strftime("%Y-%m-%d %H:%M", time_mod.localtime(m.timestamp))
                renderer.show_info(
                    f"  [{ts}] {m.action}: {m.file_path}"
                    f"{'  commit=' + m.git_commit[:8] if m.git_commit else ''}"
                )
                if m.diff_summary:
                    renderer.show_info(f"    {m.diff_summary[:100]}")
        except Exception as e:
            renderer.show_error(f"Error: {e}")

    def _handle_revert(self):
        """Handle /revert command — revert last self-modification."""
        try:
            from claude1.self_awareness.self_modifier import SelfModifier
            modifier = SelfModifier()
            result = modifier.revert_last_modification()
            renderer.show_info(result)
        except Exception as e:
            renderer.show_error(f"Revert failed: {e}")

    def _handle_reflect(self):
        """Handle /reflect command — run reflection on recent tool usage."""
        try:
            from claude1.reflection import ReflectionEngine
            engine = ReflectionEngine()

            # Gather tool calls from history
            tool_calls = []
            if self._execution_logger:
                for tc in self._execution_logger.get_recent_calls(50):
                    tool_calls.append({
                        "tool": tc.tool_name,
                        "args": tc.args_summary,
                        "result_preview": tc.result_preview,
                        "duration": tc.duration,
                    })

            result = engine.reflect_on_execution(tool_calls)

            renderer.show_info(f"Reflection ({result.total_tool_calls} tool calls analyzed):")
            for insight in result.insights:
                renderer.show_info(f"  - {insight}")
            for suggestion in result.suggestions:
                renderer.show_info(f"  Suggestion: {suggestion}")

            if result.tool_effectiveness:
                renderer.show_info("  Tool effectiveness:")
                for tool, rate in sorted(result.tool_effectiveness.items(), key=lambda x: -x[1]):
                    renderer.show_info(f"    {tool}: {rate:.0%}")
        except Exception as e:
            renderer.show_error(f"Reflection failed: {e}")

    def _handle_analytics(self):
        """Handle /analytics command — show tool usage statistics."""
        if not self._execution_logger:
            renderer.show_info("Execution logger not available.")
            return

        analytics = self._execution_logger.get_tool_analytics()
        renderer.show_info(f"Tool Analytics ({analytics.get('total_calls', 0)} total calls, {analytics.get('unique_tools', 0)} tools):")

        from rich.table import Table
        table = Table(border_style="dim")
        table.add_column("Tool", style="cyan")
        table.add_column("Calls", style="green")
        table.add_column("Failures", style="red")
        table.add_column("Failure Rate", style="yellow")
        table.add_column("Avg Duration", style="magenta")

        for tool_name, info in analytics.get("tools", {}).items():
            table.add_row(
                tool_name,
                str(info["count"]),
                str(info["failures"]),
                f"{info['failure_rate']:.0%}",
                f"{info['avg_duration']:.3f}s",
            )

        console.print(table)

        # Show patterns
        patterns = self._execution_logger.get_pattern_frequency()
        if patterns:
            renderer.show_info(f"\nCommon patterns:")
            for p in patterns[:5]:
                renderer.show_info(f"  {' -> '.join(p.sequence)} ({p.count}x, avg {p.avg_duration:.2f}s)")

    def _handle_time_budget(self, arg: str):
        """Handle /time-budget command — set time budget for next execution."""
        if not arg.strip():
            status = self._time_manager.get_status()
            if status["budget_seconds"]:
                renderer.show_info(f"Current budget: {status['budget_seconds']}s")
                renderer.show_info(f"Remaining: {status['remaining_seconds']}s" if status['remaining_seconds'] else "Not active")
            else:
                renderer.show_info("No time budget set. Usage: /time-budget <minutes>")
            return

        try:
            minutes = float(arg.strip())
            self._time_manager.total_budget_seconds = minutes * 60
            renderer.show_info(f"Time budget set: {minutes} minutes ({minutes * 60:.0f}s)")
        except ValueError:
            renderer.show_error("Usage: /time-budget <minutes>")

    def _handle_agents_command(self, arg: str) -> bool:
        """Handle /agents command and subcommands."""
        from claude1.agents.config import DEFAULT_AGENT_CONFIGS, AgentRoleConfig

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
        from claude1.agents.orchestrator import AgentOrchestrator, OrchestratorEventType
        from claude1.agents.config import DEFAULT_AGENT_CONFIGS
        from claude1.session import add_agent_run_message

        orchestrator = AgentOrchestrator(
            tool_registry=self.tool_registry,
            working_dir=self.config.working_dir,
            ollama_host=self.config.ollama_host,
            config_overrides=DEFAULT_AGENT_CONFIGS,
            task_engine=self.task_engine,
            time_manager=self._time_manager,
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
                elif event.type == OrchestratorEventType.REPLAN and event.plan:
                    renderer.show_info(f"[REPLAN] {event.content}")
                    renderer.show_agent_plan(event.plan.summary, event.plan.tasks)
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

    def _run_auto_verify(self, text: str):
        """Run auto-verify on the last assistant response (called after _process_response)."""
        from claude1.critique import run_self_critique

        # Find context (last user message)
        context = ""
        for msg in reversed(self.llm.messages):
            if msg.get("role") == "user":
                context = msg.get("content", "")
                break

        renderer.show_info("[auto-verify] Running self-critique...")
        spinner = renderer.start_thinking_spinner()
        try:
            result = run_self_critique(self.llm, text, context)
        finally:
            renderer.stop_thinking_spinner(spinner)

        if result.success and result.improvement_score > 0.1:
            renderer.show_info(f"[auto-verify] Improvement: {result.improvement_score:.0%}")
            console.print(Markdown(result.refined_output))
            # Replace the last assistant message
            for i in range(len(self.llm.messages) - 1, -1, -1):
                if self.llm.messages[i].get("role") == "assistant":
                    self.llm.messages[i]["content"] = result.refined_output
                    break
        else:
            renderer.show_info("[auto-verify] Response is good — no refinement needed.")

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

        # ── Auto-verify: critique and refine if enabled ──
        if self.config.auto_verify and text_buffer.strip():
            self._run_auto_verify(text_buffer)

        # ── Bandit update: record quality score for current model ──
        if self.config.bandit_enabled and self.llm.last_quality_result:
            try:
                from claude1.bandit import get_model_bandit
                bandit = get_model_bandit()
                bandit.update(self.config.model, self.llm.last_quality_result.score)
            except Exception:
                pass  # Don't break the REPL for bandit errors

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

        # Persist session stats
        try:
            self.session_stats.model_name = self.config.model
            self.session_stats.save_persistent()
        except Exception:
            pass

        # Clean up MCP connections
        if self._mcp_manager is not None:
            try:
                self._mcp_manager.close()
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
