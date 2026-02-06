# Claude1

A local coding assistant powered by [Ollama](https://ollama.com). Claude1 runs entirely on your machine — no API keys, no cloud, no cost.

It provides an interactive terminal REPL with streaming responses, tool calling (file read/write/edit, bash, search), session management, and a Rich-based UI.

## Features

- **7 built-in tools** — read, write, edit files, run bash commands, glob/grep search, list directories
- **Agentic tool loop** — the model chains tool calls automatically to complete multi-step tasks
- **Streaming output** — real-time markdown rendering with a thinking spinner
- **Model profiles** — per-family defaults for context window, temperature, tool support, and output format rules
- **Session management** — save, load, resume, and export conversations
- **Undo** — revert file edits with `/undo`
- **Context window management** — automatic truncation when approaching the limit
- **Retry with backoff** — resilient to transient Ollama errors
- **Pipe/stdin support** — `echo "code" | python3 main.py "review this"`
- **Print mode** — non-interactive output for scripting (`-p`)
- **CLAUDE.md project memory** — place a `CLAUDE.md` in your project root for persistent instructions
- **Git awareness** — current branch and status shown to the model
- **Syntax highlighting** — in tool results for file reads and bash output
- **Unified diffs** — colored diff display for file edits
- **Token stats** — tokens in/out, speed, session totals, and estimated API cost savings
- **Temperature & compact mode** — adjustable on the fly

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- A pulled model (default: `devstral-small-2:24b`)

### Python dependencies

```
ollama
click
rich
prompt_toolkit
```

Install them with:

```bash
pip install ollama click rich prompt_toolkit
```

## Quick Start

```bash
# Make sure Ollama is running
ollama serve

# Pull a model (if you haven't already)
ollama pull devstral-small-2:24b

# Start the REPL
python3 main.py
```

## Usage

### Interactive REPL

```bash
python3 main.py                          # start with default model
python3 main.py -m qwen3:4b             # use a different model
python3 main.py -d /path/to/project     # set working directory
python3 main.py -y                       # auto-accept all tool calls
python3 main.py -v                       # verbose/debug mode
python3 main.py -r                       # resume last session
python3 main.py --temp 0.3              # set temperature
```

### Initial prompt

```bash
python3 main.py "list the files here"
```

### Print mode (non-interactive)

```bash
python3 main.py -p "what is 2+2"
echo "def foo(): pass" | python3 main.py -p "add a docstring"
cat error.log | python3 main.py -p "explain this error"
```

### Slash commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/model <name>` | Switch model |
| `/models` | List available models |
| `/profile` | Show current model profile |
| `/clear` | Clear conversation |
| `/save <name>` | Save session |
| `/load <name>` | Load session |
| `/sessions` | List saved sessions |
| `/resume` | Resume last auto-saved session |
| `/auto` | Toggle auto-accept tool calls |
| `/temp [value]` | Get/set temperature (0.0-2.0) |
| `/compact` | Toggle compact mode |
| `/undo` | Undo last file edit/write |
| `/export [file]` | Export conversation as markdown |
| `/exit` | Exit |

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current generation |
| `Ctrl+D` | Exit |
| `Esc+Enter` | Insert newline in input |
| `Up/Down` | Navigate command history |
| `Ctrl+R` | Search command history |

## Project Memory

Create a `CLAUDE.md` file in your project root to give the model persistent instructions:

```markdown
# Project conventions
- Use snake_case for all Python functions
- Run tests with `pytest tests/`
- This project uses FastAPI
```

Global instructions can be placed at `~/.claude1/CLAUDE.md`.

## Model Profiles

Claude1 automatically detects the model family and applies appropriate defaults:

| Family | Tools | Context | Notes |
|--------|-------|---------|-------|
| Devstral | Yes | 8192 | Default model, full tool support |
| Qwen 3 | Yes | 4096 | Thinking suppression, tuned sampling |
| Gemma 3 | No | 4096 | Text-only mode |
| Llama 3.2 | Yes | 4096 | Tool support (except 1B variant) |
| DeepSeek R1 | No | 4096 | Reasoning model, text-only |
| Cogito | Yes | 4096 | Tool support |
| Mistral | Yes | 4096 | Tool support |

Switch models at runtime with `/model <name>` — the profile updates automatically.

## Architecture

```
main.py              CLI entry point (Click)
config.py            Configuration constants and AppConfig
model_profiles.py    Per-model-family profiles and defaults
repl.py              REPL orchestration
llm.py               LLM interface with agentic tool loop
system_prompt.py     Dynamic system prompt (git, CLAUDE.md, profiles)
session.py           Session save/load/export
stats.py             Token/cost tracking
undo.py              File edit undo stack
tools/
  __init__.py        Tool registry
  base.py            Abstract base tool
  file_tools.py      read_file, write_file, edit_file
  bash_tool.py       bash command execution
  search_tools.py    glob_search, grep_search, list_dir
ui/
  renderer.py        Rich-based terminal rendering
  prompts.py         prompt_toolkit input handling
```

## Data Storage

All persistent data is stored in `~/.claude1/`:

- `history` — command history
- `sessions/` — saved conversation sessions
- `CLAUDE.md` — global user instructions

## License

MIT
