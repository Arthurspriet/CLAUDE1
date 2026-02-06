"""Per-model-family profile system for consistent output formatting and tool gating."""

from dataclasses import dataclass


# ── Format-enforcement string constants ──────────────────────────────────────

_STANDARD_FORMAT_RULES = """
- Always use fenced code blocks with language tags (```python, ```bash, etc.)
- Lead with a one-sentence explanation before code
- Use proper markdown formatting (headers, lists, bold)
- No filler phrases ("Sure!", "Of course!", "Let me help you with that!")
- Show full file paths when referencing files
""".strip()

_NO_TOOLS_RULES = """
- You do NOT have access to any tools. Do not attempt tool calls.
- Provide code for the user to copy and run manually.
- When showing file changes, use clear before/after code blocks with file paths.
""".strip()

_QWEN_THINKING_RULES = """
- Do NOT output your chain-of-thought or thinking process.
- Suppress any visible <think> blocks — go straight to the answer.
""".strip()

_REASONING_MODEL_RULES = """
- You may use <think> blocks for internal reasoning.
- After reasoning, provide a clean, structured final answer.
- Keep the final answer concise — the thinking block is for you, the answer is for the user.
""".strip()

_SMALL_MODEL_RULES = """
- Focus on precision over breadth. Answer exactly what was asked.
- Do not attempt complex multi-step plans or large refactors.
- Keep code snippets short and targeted.
- If a task is too complex, say so and suggest breaking it down.
""".strip()

_PLANNING_RULES = """
- Before executing tools, briefly state your plan:
  1. What you are trying to accomplish
  2. Which tools you will use and why
  3. Any risks or edge cases to watch for
- Keep the plan to 2-4 sentences. Be concrete, not generic.
""".strip()


# ── Behavioral guidance constants ─────────────────────────────────────────────

_TOOL_CALLING_BEHAVIOR = """
- Call one tool at a time and wait for its result before deciding the next action.
- Never assume file contents — always read before editing.
- If a tool call fails, explain the error briefly, then adjust your approach and retry.
""".strip()

_WEAK_TOOL_CALLING_BEHAVIOR = """
- Call one tool at a time. Wait for the result.
- ALWAYS use read_file before edit_file. No exceptions.
- Copy old_string exactly from read_file output, including all whitespace.
- If edit_file fails, re-read the file and retry with corrected old_string.
""".strip()

_NO_OVER_GENERATION = """
- Stop after answering the user's question. Do not add unsolicited suggestions or follow-up ideas.
""".strip()


# ── ModelProfile dataclass ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelProfile:
    """Immutable profile describing a model family's capabilities and defaults."""

    family: str
    display_name: str
    supports_tools: bool = True
    num_ctx: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    behavioral_rules: str = ""
    system_prompt_suffix: str = _STANDARD_FORMAT_RULES


# ── Profile registry ─────────────────────────────────────────────────────────

PROFILES: dict[str, ModelProfile] = {
    "devstral": ModelProfile(
        family="devstral",
        display_name="Devstral",
        supports_tools=True,
        num_ctx=8192,
        behavioral_rules=_TOOL_CALLING_BEHAVIOR,
    ),
    "qwen3": ModelProfile(
        family="qwen3",
        display_name="Qwen 3",
        supports_tools=True,
        temperature=0.7,
        top_p=0.8,
        repeat_penalty=1.05,
        behavioral_rules=_TOOL_CALLING_BEHAVIOR + "\n" + _NO_OVER_GENERATION,
        system_prompt_suffix=_STANDARD_FORMAT_RULES + "\n" + _QWEN_THINKING_RULES,
    ),
    "gemma3": ModelProfile(
        family="gemma3",
        display_name="Gemma 3",
        supports_tools=False,
        temperature=0.7,
        system_prompt_suffix=_STANDARD_FORMAT_RULES + "\n" + _NO_TOOLS_RULES,
    ),
    "llama3.2": ModelProfile(
        family="llama3.2",
        display_name="Llama 3.2",
        supports_tools=True,
        num_ctx=4096,
        behavioral_rules=_WEAK_TOOL_CALLING_BEHAVIOR,
    ),
    "llama3.2:1b": ModelProfile(
        family="llama3.2:1b",
        display_name="Llama 3.2 1B",
        supports_tools=False,
        num_ctx=2048,
        system_prompt_suffix=_STANDARD_FORMAT_RULES + "\n" + _NO_TOOLS_RULES + "\n" + _SMALL_MODEL_RULES,
    ),
    "deepseek-r1": ModelProfile(
        family="deepseek-r1",
        display_name="DeepSeek R1",
        supports_tools=False,
        temperature=0.6,
        system_prompt_suffix=_STANDARD_FORMAT_RULES + "\n" + _NO_TOOLS_RULES + "\n" + _REASONING_MODEL_RULES,
    ),
    "cogito": ModelProfile(
        family="cogito",
        display_name="Cogito",
        supports_tools=True,
        behavioral_rules=_TOOL_CALLING_BEHAVIOR,
    ),
    "mistral": ModelProfile(
        family="mistral",
        display_name="Mistral",
        supports_tools=True,
        behavioral_rules=_TOOL_CALLING_BEHAVIOR,
    ),
    "ministral": ModelProfile(
        family="ministral",
        display_name="Ministral",
        supports_tools=False,
        system_prompt_suffix=_STANDARD_FORMAT_RULES + "\n" + _NO_TOOLS_RULES,
    ),
    "llava": ModelProfile(
        family="llava",
        display_name="LLaVA",
        supports_tools=False,
        num_ctx=4096,
        system_prompt_suffix=_STANDARD_FORMAT_RULES + "\n" + _NO_TOOLS_RULES,
    ),
    "olmo": ModelProfile(
        family="olmo",
        display_name="OLMo",
        supports_tools=False,
        num_ctx=4096,
        system_prompt_suffix=_STANDARD_FORMAT_RULES + "\n" + _NO_TOOLS_RULES + "\n" + _SMALL_MODEL_RULES,
    ),
    # HuggingFace model families
    "meta-llama": ModelProfile(
        family="meta-llama",
        display_name="Meta Llama (HF)",
        supports_tools=True,
        num_ctx=8192,
        behavioral_rules=_TOOL_CALLING_BEHAVIOR,
    ),
    "mistralai": ModelProfile(
        family="mistralai",
        display_name="Mistral AI (HF)",
        supports_tools=True,
        num_ctx=8192,
        behavioral_rules=_TOOL_CALLING_BEHAVIOR,
    ),
    "Qwen": ModelProfile(
        family="Qwen",
        display_name="Qwen (HF)",
        supports_tools=True,
        num_ctx=8192,
        temperature=0.7,
        top_p=0.8,
        behavioral_rules=_TOOL_CALLING_BEHAVIOR + "\n" + _NO_OVER_GENERATION,
        system_prompt_suffix=_STANDARD_FORMAT_RULES + "\n" + _QWEN_THINKING_RULES,
    ),
    "microsoft": ModelProfile(
        family="microsoft",
        display_name="Microsoft (HF)",
        supports_tools=True,
        num_ctx=8192,
        behavioral_rules=_TOOL_CALLING_BEHAVIOR,
    ),
}

DEFAULT_PROFILE = ModelProfile(
    family="default",
    display_name="Default",
    supports_tools=True,
    num_ctx=4096,
    behavioral_rules=_TOOL_CALLING_BEHAVIOR,
)


# ── Lookup function ──────────────────────────────────────────────────────────

def get_profile(model_name: str, ollama_family: str | None = None) -> ModelProfile:
    """Resolve a ModelProfile for the given model name.

    Fallback chain:
      1. Strip hf: prefix for HuggingFace models
      2. Exact model name match (handles variant overrides like 'llama3.2:1b')
      3. Ollama family metadata from ollama.show()
      4. HF org-name fallback ('meta-llama/Llama-3-8B' -> match 'meta-llama')
      5. Progressive prefix match ('devstral-small-2:24b' -> 'devstral')
      6. DEFAULT_PROFILE
    """
    # 1. Strip hf: prefix
    lookup_name = model_name
    if lookup_name.startswith("hf:"):
        lookup_name = lookup_name[3:]

    # 2. Exact model name match
    if lookup_name in PROFILES:
        return PROFILES[lookup_name]

    # 3. Ollama family metadata
    if ollama_family and ollama_family in PROFILES:
        return PROFILES[ollama_family]

    # 4. HF org-name fallback (e.g. 'meta-llama/Llama-3-8B' -> try 'meta-llama')
    if "/" in lookup_name:
        org = lookup_name.split("/", 1)[0]
        if org in PROFILES:
            return PROFILES[org]

    # 5. Progressive prefix match — try longest prefix first
    base = lookup_name.split(":")[0]
    for key in sorted(PROFILES.keys(), key=len, reverse=True):
        if base.startswith(key) or lookup_name.startswith(key):
            return PROFILES[key]

    # 6. Fallback
    return DEFAULT_PROFILE


# ── Display helper ───────────────────────────────────────────────────────────

def format_profile_info(profile: ModelProfile) -> str:
    """Return a multi-line string describing the profile for /profile display."""
    lines = [
        f"Profile: {profile.display_name} (family: {profile.family})",
        f"  Tools:          {'yes' if profile.supports_tools else 'no'}",
        f"  Context window: {profile.num_ctx}",
    ]
    if profile.temperature is not None:
        lines.append(f"  Temperature:    {profile.temperature}")
    if profile.top_p is not None:
        lines.append(f"  Top-p:          {profile.top_p}")
    if profile.top_k is not None:
        lines.append(f"  Top-k:          {profile.top_k}")
    if profile.repeat_penalty is not None:
        lines.append(f"  Repeat penalty: {profile.repeat_penalty}")

    suffix_preview = profile.system_prompt_suffix[:120]
    if len(profile.system_prompt_suffix) > 120:
        suffix_preview += "..."
    lines.append(f"  Format rules:   {suffix_preview}")

    return "\n".join(lines)
