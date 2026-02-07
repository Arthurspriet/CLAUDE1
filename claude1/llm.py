"""LLM interface with agentic tool-calling loop."""

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import ollama

import logging

from claude1.config import AppConfig, HF_TOKEN, HF_ENDPOINT, MAX_TOOL_ITERATIONS, MAX_RETRIES, RETRY_BASE_DELAY, CONTEXT_WINDOW_RESERVE, HF_CLIENT_TIMEOUT, HF_STREAM_CHUNK_TIMEOUT
from claude1.logging_config import log_tool_execution

logger = logging.getLogger("claude1.llm")
from claude1.response_quality import (
    evaluate_response,
    ResponseQualityResult,
    MAX_REFUSAL_RETRIES,
    MAX_QUALITY_RETRIES,
    MIN_QUALITY_SCORE,
)
from claude1.system_prompt import build_system_prompt
from claude1.tools import ToolRegistry

# Threshold for detecting text dumps (characters without any tool call)
TEXT_DUMP_THRESHOLD = 3000
# Maximum number of nudges to prevent infinite nudge loops
MAX_NUDGES = 2
# The nudge message injected when the model dumps text instead of using tools
_NUDGE_MESSAGE = (
    "You just generated a long text response without using any tools. "
    "Remember: you are an AGENT. Do not describe code — use write_file to create files, "
    "edit_file to modify them, and bash to run commands. "
    "Now use your tools to actually implement what you described."
)
# Message injected when the model refuses a request
_REFUSAL_NUDGE = (
    "Your previous response was a refusal. The user's request is valid and within scope. "
    "Please try again and provide a helpful, substantive response. "
    "Focus on addressing the request directly."
)
# Message injected when the response quality is too low
_QUALITY_NUDGE = (
    "Your previous response was too short or lacked substance. "
    "Please provide a more complete, well-structured answer. "
    "Include relevant details, examples, or code as appropriate."
)

# Guarded import for huggingface_hub
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    InferenceClient = None
    HF_AVAILABLE = False


@dataclass
class StreamChunk:
    """A chunk of output from the agentic loop."""

    type: str  # "text", "tool_call", "tool_result", "done", "error", "thinking_start", "thinking_stop", "stats", "debug"
    content: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)


class LLMInterface:
    """Manages conversation history and the agentic tool-calling loop."""

    def __init__(
        self,
        config: AppConfig,
        tool_registry: ToolRegistry,
        confirm_callback: Callable[[str, dict], bool] | None = None,
    ):
        self.config = config
        self.tools = tool_registry
        self.confirm_callback = confirm_callback
        self.messages: list[dict[str, Any]] = []
        self._cancelled = False
        self._last_prompt_eval_count: int | None = None
        self.active_tool_filter: list[str] | None = None
        self._hf_client: Any = None
        self._nudge_count: int = 0  # Track nudges to avoid infinite loops
        self._refusal_retry_count: int = 0  # Track refusal retries
        self._quality_retry_count: int = 0  # Track quality retries
        self.last_quality_result: ResponseQualityResult | None = None  # Last quality eval

        # Initialize system prompt
        self._set_system_prompt()

    def _get_hf_client(self):
        """Lazily initialize and return the HuggingFace InferenceClient."""
        if self._hf_client is None:
            if not HF_AVAILABLE:
                raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface_hub")
            kwargs: dict[str, Any] = {"timeout": HF_CLIENT_TIMEOUT}
            # Local TGI endpoint takes priority
            endpoint = self.config.hf_endpoint or HF_ENDPOINT
            if endpoint:
                kwargs["base_url"] = endpoint
            token = HF_TOKEN
            if token:
                kwargs["token"] = token
            self._hf_client = InferenceClient(**kwargs)
        return self._hf_client

    def _set_system_prompt(self):
        """Set or update the system prompt."""
        tool_names = [t.name for t in self.tools.all_tools()]
        prompt = build_system_prompt(
            self.config.working_dir, self.config.model,
            compact=self.config.compact, profile=self.config.profile,
            planning=self.config.planning,
            tool_names=tool_names,
        )
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": prompt}
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def update_model(self, model: str):
        """Switch to a different model."""
        from claude1.config import parse_model_spec
        provider, model_id = parse_model_spec(model)
        self.config.model = model
        self.config.provider = provider
        # Reset HF client so it gets re-initialized with new settings
        self._hf_client = None
        self._set_system_prompt()

    def clear_history(self):
        """Clear conversation history, keeping only the system prompt."""
        self.messages = [self.messages[0]] if self.messages else []
        self._set_system_prompt()

    def cancel(self):
        """Signal cancellation of current generation."""
        self._cancelled = True

    def set_tool_filter(self, tools: list[str] | None):
        """Restrict available tools to only the named ones."""
        self.active_tool_filter = tools

    def clear_tool_filter(self):
        """Restore full tool set."""
        self.active_tool_filter = None

    def get_context_usage(self) -> dict:
        """Return context window usage information."""
        estimated = self._estimate_tokens()
        num_ctx = self.config.num_ctx
        reserve = CONTEXT_WINDOW_RESERVE
        usable = num_ctx - reserve

        # Per-role breakdown
        encoder = self._get_tokenizer()
        role_counts: dict[str, int] = {}
        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                if encoder is not None:
                    tokens = len(encoder.encode(content))
                else:
                    tokens = int(len(content) / 3.5)
            else:
                tokens = 0
            role_counts[role] = role_counts.get(role, 0) + tokens

        return {
            "estimated_tokens": estimated,
            "num_ctx": num_ctx,
            "reserve": reserve,
            "usable": usable,
            "message_count": len(self.messages),
            "role_breakdown": role_counts,
            "usage_pct": (estimated / usable * 100) if usable > 0 else 0,
        }

    def _estimate_tokens(self) -> int:
        """Estimate token count for current messages.

        Uses actual prompt_eval_count from the last Ollama response when available.
        Otherwise falls back to tiktoken (if installed) or a character-based heuristic
        calibrated for code (1 token ≈ 3.5 chars, not 4).
        """
        if self._last_prompt_eval_count is not None:
            return self._last_prompt_eval_count

        # Try tiktoken for more accurate estimation
        encoder = self._get_tokenizer()
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if encoder is not None:
                    total += len(encoder.encode(content))
                else:
                    # Calibrated heuristic: code averages ~3.5 chars/token
                    total += int(len(content) / 3.5)
            # Account for tool call args
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                args = tc.get("function", {}).get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                if encoder is not None:
                    total += len(encoder.encode(args_str))
                else:
                    total += int(len(args_str) / 3.5)
        return total

    def _get_tokenizer(self):
        """Get a cached tokenizer. Returns tiktoken encoder or None."""
        if not hasattr(self, "_tokenizer_cache"):
            try:
                import tiktoken
                # cl100k_base works well for most models
                self._tokenizer_cache = tiktoken.get_encoding("cl100k_base")
            except (ImportError, Exception):
                self._tokenizer_cache = None
        return self._tokenizer_cache

    def _truncate_if_needed(self):
        """Truncate conversation using priority-based retention.

        Priority order:
        1. System prompt (always kept)
        2. Latest user message + all subsequent messages (always kept)
        3. Recent tool-call/result pairs (kept intact, not split)
        4. Older messages (progressively summarized with decreasing detail)
        """
        estimated = self._estimate_tokens()
        limit = int((self.config.num_ctx - CONTEXT_WINDOW_RESERVE) * 0.8)

        if estimated <= limit:
            return

        if len(self.messages) <= 4:
            return

        system = self.messages[0]
        rest = self.messages[1:]

        # Find the last user message index (in rest[])
        last_user_idx = -1
        for i in range(len(rest) - 1, -1, -1):
            if rest[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx < 0:
            return

        # Split into: older messages (before last user) + recent (last user + after)
        recent = rest[last_user_idx:]
        older = rest[:last_user_idx]

        if not older:
            return

        # Group older messages into tool-call chains (assistant with tool_calls + tool results)
        groups: list[list[dict]] = []
        current_group: list[dict] = []

        for msg in older:
            role = msg.get("role", "")
            has_tool_calls = bool(msg.get("tool_calls"))

            if role == "tool":
                # Tool result — attach to current group
                current_group.append(msg)
            elif role == "assistant" and has_tool_calls:
                # Start of a tool-call chain — flush previous group
                if current_group:
                    groups.append(current_group)
                current_group = [msg]
            else:
                # Regular message
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([msg])

        if current_group:
            groups.append(current_group)

        # Build summarized version: keep recent groups intact, summarize oldest
        # Strategy: keep the most recent N groups that fit, summarize the rest
        kept_groups: list[list[dict]] = []
        summary_parts: list[str] = []

        # Estimate tokens for recent + system
        encoder = self._get_tokenizer()
        system_tokens = self._count_msg_tokens(system, encoder)
        recent_tokens = sum(self._count_msg_tokens(m, encoder) for m in recent)
        budget = limit - system_tokens - recent_tokens - 200  # 200 for summary overhead

        # Walk groups from newest to oldest, keep what fits
        for group in reversed(groups):
            group_tokens = sum(self._count_msg_tokens(m, encoder) for m in group)
            if budget >= group_tokens:
                kept_groups.insert(0, group)
                budget -= group_tokens
            else:
                # Summarize this group
                for msg in group:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        # Progressive summarization: shorter snippets for older messages
                        max_len = max(80, min(300, budget // 2))
                        snippet = content[:max_len] + "..." if len(content) > max_len else content
                        # Extract key info (file paths, decisions)
                        summary_parts.append(f"[{role}]: {snippet}")

        # Assemble final messages
        result_messages = [system]

        if summary_parts:
            summary = "[Earlier conversation summary]\n" + "\n".join(summary_parts)
            result_messages.append({"role": "user", "content": summary})

        for group in kept_groups:
            result_messages.extend(group)

        result_messages.extend(recent)
        self.messages = result_messages

    def _count_msg_tokens(self, msg: dict, encoder) -> int:
        """Count tokens in a single message."""
        content = msg.get("content", "")
        if not isinstance(content, str):
            return 0
        if encoder is not None:
            return len(encoder.encode(content))
        return int(len(content) / 3.5)

    def _build_plan_prompt(self, tool_calls: list[dict]) -> str:
        """Construct a prompt asking the model to explain its plan for the pending tool calls."""
        lines = ["You are about to execute these tools:"]
        for i, tc in enumerate(tool_calls, 1):
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            # Build a concise argument summary
            arg_parts = []
            for k, v in args.items():
                val = str(v)
                if len(val) > 60:
                    val = val[:60] + "..."
                arg_parts.append(f'{k}="{val}"')
            arg_str = ", ".join(arg_parts)
            lines.append(f"  {i}. {name}({arg_str})")
        lines.append("")
        lines.append("Briefly explain your plan: what are you trying to accomplish, why these tools in this order, and any risks.")
        return "\n".join(lines)

    def _run_planning_call(self, plan_prompt: str) -> str | None:
        """Make a non-streaming API call to get the model's plan. Returns plan text or None.

        For HuggingFace, the call is guarded by the client-level timeout set in
        _get_hf_client, so it won't block indefinitely.
        """
        plan_messages = list(self.messages) + [{"role": "user", "content": plan_prompt}]

        try:
            if self.config.provider == "huggingface":
                client = self._get_hf_client()
                response = client.chat_completion(
                    model=self.config.model[3:],  # strip hf: prefix
                    messages=plan_messages,
                    max_tokens=1024,
                    stream=False,
                )
                content = response.choices[0].message.content if response.choices else ""
            else:
                response = ollama.chat(
                    model=self.config.model,
                    messages=plan_messages,
                    stream=False,
                    options={"num_ctx": self.config.num_ctx},
                )
                content = response.get("message", {}).get("content", "") if isinstance(response, dict) else getattr(getattr(response, "message", None), "content", "")
            return content.strip() if content and content.strip() else None
        except Exception:
            return None

    def _make_ollama_api_call(self, tool_defs: list[dict], options: dict) -> Any:
        """Make a streaming API call to Ollama. Returns the stream iterator."""
        return ollama.chat(
            model=self.config.model,
            messages=self.messages,
            tools=tool_defs,
            stream=True,
            options=options,
        )

    def _make_hf_api_call(self, tool_defs: list[dict]) -> Any:
        """Make a streaming API call to HuggingFace. Returns the stream iterator."""
        client = self._get_hf_client()
        kwargs: dict[str, Any] = {
            "model": self.config.model[3:],  # strip hf: prefix
            "messages": self.messages,
            "max_tokens": 4096,
            "stream": True,
        }
        if tool_defs:
            kwargs["tools"] = tool_defs
        # Temperature
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        elif self.config.profile and self.config.profile.temperature is not None:
            kwargs["temperature"] = self.config.profile.temperature
        return client.chat_completion(**kwargs)

    def _process_ollama_stream(self, stream) -> Iterator[tuple[str, list[dict], bool]]:
        """Process Ollama stream chunks. Yields (token, tool_calls_batch, is_done) and stats side-effects."""
        for chunk in stream:
            if self._cancelled:
                yield ("", [], False)
                return

            msg = chunk.get("message", {})
            token = msg.get("content", "")
            tc_batch = msg.get("tool_calls", [])

            is_done = chunk.get("done", False)
            if is_done:
                prompt_eval_count = chunk.get("prompt_eval_count", 0)
                eval_count = chunk.get("eval_count", 0)
                eval_duration = chunk.get("eval_duration", 0)
                self._last_prompt_eval_count = prompt_eval_count
                self._ollama_stats = {
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": eval_count,
                    "eval_duration": eval_duration,
                }

            yield (token, tc_batch, is_done)

    @staticmethod
    def _iter_with_timeout(stream, timeout: float):
        """Wrap a blocking stream iterator so that each chunk must arrive within *timeout* seconds.

        Uses a background thread to read from the stream and a queue to shuttle
        chunks back to the caller.  Raises TimeoutError if no chunk arrives in time.
        """
        _SENTINEL = object()
        q: queue.Queue = queue.Queue()

        def _reader():
            try:
                for item in stream:
                    q.put(item)
                q.put(_SENTINEL)
            except Exception as exc:
                q.put(exc)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        while True:
            try:
                item = q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(
                    f"HuggingFace stream stalled: no data received for {timeout}s"
                )
            if item is _SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def _process_hf_stream(self, stream) -> Iterator[tuple[str, list[dict], bool]]:
        """Process HuggingFace stream chunks with tool call delta accumulation.

        HF streams tool calls incrementally: name in one chunk, args across several.
        We accumulate by tool call index and yield completed tool calls at the end.

        Each chunk must arrive within HF_STREAM_CHUNK_TIMEOUT seconds or a
        TimeoutError is raised (caught by the caller's except block).
        """
        # Accumulators for tool call deltas indexed by position
        tc_accum: dict[int, dict] = {}

        for chunk in self._iter_with_timeout(stream, HF_STREAM_CHUNK_TIMEOUT):
            if self._cancelled:
                yield ("", [], False)
                return

            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue

            delta = choice.delta
            token = delta.content or "" if delta else ""

            # Accumulate tool call deltas
            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index if hasattr(tc_delta, "index") and tc_delta.index is not None else 0
                    if idx not in tc_accum:
                        tc_accum[idx] = {"id": "", "name": "", "arguments": ""}

                    if hasattr(tc_delta, "id") and tc_delta.id:
                        tc_accum[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_accum[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_accum[idx]["arguments"] += tc_delta.function.arguments

            is_done = choice.finish_reason is not None
            yield (token, [], is_done)

        # After stream ends, convert accumulated tool calls into standard format
        if tc_accum:
            completed_calls = []
            for idx in sorted(tc_accum.keys()):
                tc = tc_accum[idx]
                call_id = tc["id"] or f"call_{uuid.uuid4().hex[:8]}"
                args_str = tc["arguments"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                completed_calls.append({
                    "id": call_id,
                    "function": {
                        "name": tc["name"],
                        "arguments": args,
                    },
                })
            # Yield the completed tool calls
            yield ("", completed_calls, True)

    def send_message(self, user_input: str) -> Iterator[StreamChunk]:
        """Send a user message and yield streaming chunks through the agentic loop."""
        self._cancelled = False
        self._nudge_count = 0  # Reset nudge counter for each new user message
        self._refusal_retry_count = 0  # Reset refusal counter
        self._quality_retry_count = 0  # Reset quality counter
        self.last_quality_result = None
        self.messages.append({"role": "user", "content": user_input})

        # Truncate if context is getting large
        self._truncate_if_needed()

        is_hf = self.config.provider == "huggingface"

        # Skip tool definitions if profile says tools aren't supported
        profile = self.config.profile
        if profile and not profile.supports_tools:
            tool_defs = []
        else:
            tool_defs = self.tools.tool_definitions()
            # Apply tool filter if active (e.g. during skill execution)
            if self.active_tool_filter is not None:
                allowed = set(self.active_tool_filter)
                tool_defs = [t for t in tool_defs if t.get("function", {}).get("name") in allowed]

        for iteration in range(MAX_TOOL_ITERATIONS):
            if self._cancelled:
                yield StreamChunk(type="done", content="[cancelled]")
                return

            # Debug info before API call
            if self.config.verbose:
                yield StreamChunk(
                    type="debug",
                    content=f"API call: {len(self.messages)} messages, model={self.config.model}, provider={self.config.provider}, tools={len(tool_defs)}"
                )

            # Emit thinking start
            yield StreamChunk(type="thinking_start")

            # Build Ollama options dict (only used for Ollama provider)
            options: dict[str, Any] = {"num_ctx": self.config.num_ctx}
            if not is_hf:
                if profile:
                    if profile.top_p is not None:
                        options["top_p"] = profile.top_p
                    if profile.top_k is not None:
                        options["top_k"] = profile.top_k
                    if profile.repeat_penalty is not None:
                        options["repeat_penalty"] = profile.repeat_penalty
                if self.config.temperature is not None:
                    options["temperature"] = self.config.temperature
                elif profile and profile.temperature is not None:
                    options["temperature"] = profile.temperature

            # Retry loop with exponential backoff
            stream = None
            self._ollama_stats = None
            for attempt in range(MAX_RETRIES):
                try:
                    if is_hf:
                        stream = self._make_hf_api_call(tool_defs)
                    else:
                        stream = self._make_ollama_api_call(tool_defs, options)
                    break
                except Exception as e:
                    is_retryable = isinstance(e, (ConnectionError, TimeoutError, OSError))
                    if is_hf:
                        # Retry on HF-specific transient errors
                        try:
                            from huggingface_hub import errors as hf_errors
                            is_retryable = is_retryable or isinstance(
                                e, (hf_errors.InferenceTimeoutError, hf_errors.OverloadedError, hf_errors.HfHubHTTPError)
                            )
                        except ImportError:
                            pass
                    else:
                        is_retryable = is_retryable or isinstance(e, ollama.ResponseError)
                    if is_retryable and attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        yield StreamChunk(type="text", content=f"\n[Retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s...]\n")
                        time.sleep(delay)
                    else:
                        yield StreamChunk(type="thinking_stop")
                        yield StreamChunk(type="error", content=f"Failed after {MAX_RETRIES} retries: {e}" if is_retryable else f"Error: {e}")
                        return

            if stream is None:
                yield StreamChunk(type="thinking_stop")
                yield StreamChunk(type="error", content="Failed to get response from model")
                return

            # Accumulate the response
            full_text = ""
            tool_calls = []
            thinking_stopped = False

            try:
                # Use provider-specific stream processor
                processor = self._process_hf_stream(stream) if is_hf else self._process_ollama_stream(stream)

                for token, tc_batch, is_done in processor:
                    if self._cancelled:
                        if not thinking_stopped:
                            yield StreamChunk(type="thinking_stop")
                        yield StreamChunk(type="done", content="[cancelled]")
                        return

                    # Stream text content
                    if token:
                        if not thinking_stopped:
                            yield StreamChunk(type="thinking_stop")
                            thinking_stopped = True
                        full_text += token
                        yield StreamChunk(type="text", content=token)

                    # Collect tool calls
                    if tc_batch:
                        if not thinking_stopped:
                            yield StreamChunk(type="thinking_stop")
                            thinking_stopped = True
                        tool_calls.extend(tc_batch)

                    # Emit stats on done (Ollama only — HF doesn't provide these)
                    if is_done and not is_hf and self._ollama_stats:
                        stats = self._ollama_stats
                        if stats.get("prompt_eval_count") or stats.get("eval_count"):
                            yield StreamChunk(type="stats", content=json.dumps(stats))

            except Exception as e:
                if not thinking_stopped:
                    yield StreamChunk(type="thinking_stop")
                yield StreamChunk(type="error", content=f"Streaming error: {e}")
                return

            if not thinking_stopped:
                yield StreamChunk(type="thinking_stop")

            # Build the assistant message to add to history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": full_text}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            # If no tool calls, check if this was a text dump that should have used tools
            if not tool_calls:
                # Detect text dumps: long text response with no tool calls
                # Only nudge if tools are available and we haven't nudged too many times
                if (
                    tool_defs
                    and len(full_text) > TEXT_DUMP_THRESHOLD
                    and self._nudge_count < MAX_NUDGES
                ):
                    self._nudge_count += 1
                    # Inject a nudge to get the model back on track
                    self.messages.append({
                        "role": "user",
                        "content": _NUDGE_MESSAGE,
                    })
                    yield StreamChunk(
                        type="debug",
                        content=f"Text dump detected ({len(full_text)} chars, no tools). Nudging model to use tools. (nudge {self._nudge_count}/{MAX_NUDGES})",
                    )
                    # Continue the loop — the model will try again
                    continue

                # ── Response quality checks (ported from deep_thinker) ──
                # Find the original user prompt for echo detection
                original_prompt = user_input
                quality = evaluate_response(full_text, original_prompt)
                self.last_quality_result = quality

                # Check for refusal — retry with a nudge
                if quality.refusal and self._refusal_retry_count < MAX_REFUSAL_RETRIES:
                    self._refusal_retry_count += 1
                    self.messages.append({
                        "role": "user",
                        "content": _REFUSAL_NUDGE,
                    })
                    yield StreamChunk(
                        type="debug",
                        content=f"Refusal detected: '{quality.refusal}'. Retrying ({self._refusal_retry_count}/{MAX_REFUSAL_RETRIES})...",
                    )
                    continue

                # Check for prompt echo — retry
                if quality.echo and self._quality_retry_count < MAX_QUALITY_RETRIES:
                    self._quality_retry_count += 1
                    self.messages.append({
                        "role": "user",
                        "content": _QUALITY_NUDGE,
                    })
                    yield StreamChunk(
                        type="debug",
                        content=f"Prompt echo detected. Retrying ({self._quality_retry_count}/{MAX_QUALITY_RETRIES})...",
                    )
                    continue

                # Check for low quality — retry once
                if quality.score < MIN_QUALITY_SCORE and self._quality_retry_count < MAX_QUALITY_RETRIES:
                    self._quality_retry_count += 1
                    self.messages.append({
                        "role": "user",
                        "content": _QUALITY_NUDGE,
                    })
                    yield StreamChunk(
                        type="debug",
                        content=f"Low quality response ({quality.score:.2f} < {MIN_QUALITY_SCORE}). Retrying ({self._quality_retry_count}/{MAX_QUALITY_RETRIES})...",
                    )
                    continue

                # Emit quality info in verbose mode
                if self.config.verbose and quality.score > 0:
                    yield StreamChunk(
                        type="debug",
                        content=f"Response quality: {quality.score:.2f}" + (f" | {', '.join(quality.issues)}" if quality.issues else ""),
                    )

                yield StreamChunk(type="done")
                return

            # Planning step: ask the model to explain its plan before executing tools
            if self.config.planning:
                plan_prompt = self._build_plan_prompt(tool_calls)
                plan_text = self._run_planning_call(plan_prompt)
                if plan_text:
                    yield StreamChunk(type="plan", content=plan_text)

            # Execute tool calls
            for tc in tool_calls:
                if self._cancelled:
                    yield StreamChunk(type="done", content="[cancelled]")
                    return

                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                tool_args = func.get("arguments", {})
                tool_call_id = tc.get("id", "")

                # Ensure tool_args is a dict (with JSON repair for small models)
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        from claude1.json_repair import repair_json
                        tool_args = repair_json(tool_args)
                        if not isinstance(tool_args, dict):
                            tool_args = {"raw": tool_args}

                yield StreamChunk(
                    type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                )

                tool = self.tools.get(tool_name)
                tool_elapsed = 0.0
                if tool is None:
                    result = f"Error: Unknown tool '{tool_name}'"
                else:
                    # Check confirmation
                    if tool.requires_confirmation and self.confirm_callback:
                        if not self.confirm_callback(tool_name, tool_args):
                            result = "Tool execution denied by user."
                        else:
                            tool_start = time.time()
                            result = tool.execute(**tool_args)
                            tool_elapsed = time.time() - tool_start
                            if self.config.verbose:
                                yield StreamChunk(
                                    type="debug",
                                    content=f"Tool {tool_name} executed in {tool_elapsed:.2f}s"
                                )
                    else:
                        tool_start = time.time()
                        result = tool.execute(**tool_args)
                        tool_elapsed = time.time() - tool_start
                        if self.config.verbose:
                            yield StreamChunk(
                                type="debug",
                                content=f"Tool {tool_name} executed in {tool_elapsed:.2f}s"
                            )

                # Audit log for tool execution
                if tool is not None and result != "Tool execution denied by user.":
                    log_tool_execution(tool_name, tool_args, result, tool_elapsed)

                yield StreamChunk(
                    type="tool_result",
                    tool_name=tool_name,
                    content=result,
                    tool_args=tool_args,
                )

                # Add tool result to conversation
                # HF/OpenAI requires tool_call_id on tool result messages
                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": result,
                }
                if is_hf and tool_call_id:
                    tool_result_msg["tool_call_id"] = tool_call_id
                self.messages.append(tool_result_msg)

            # Loop back for the model to process tool results

        # Exceeded max iterations
        yield StreamChunk(
            type="error",
            content=f"Reached maximum tool iterations ({MAX_TOOL_ITERATIONS}). Stopping.",
        )
