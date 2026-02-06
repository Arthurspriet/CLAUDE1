"""LLM interface with agentic tool-calling loop."""

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import ollama

from config import AppConfig, HF_TOKEN, HF_ENDPOINT, MAX_TOOL_ITERATIONS, MAX_RETRIES, RETRY_BASE_DELAY, CONTEXT_WINDOW_RESERVE, HF_CLIENT_TIMEOUT, HF_STREAM_CHUNK_TIMEOUT
from system_prompt import build_system_prompt
from tools import ToolRegistry

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
        prompt = build_system_prompt(
            self.config.working_dir, self.config.model,
            compact=self.config.compact, profile=self.config.profile,
            planning=self.config.planning,
        )
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": prompt}
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def update_model(self, model: str):
        """Switch to a different model."""
        from config import parse_model_spec
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
        role_counts: dict[str, int] = {}
        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tokens = len(content) // 4 if isinstance(content, str) else 0
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
        """Estimate token count for current messages."""
        if self._last_prompt_eval_count is not None:
            return self._last_prompt_eval_count
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
        return total

    def _truncate_if_needed(self):
        """Truncate conversation if approaching context window limit."""
        estimated = self._estimate_tokens()
        limit = int((self.config.num_ctx - CONTEXT_WINDOW_RESERVE) * 0.8)

        if estimated <= limit:
            return

        # Keep system prompt (index 0) + last 6 messages
        if len(self.messages) <= 7:
            return

        system = self.messages[0]
        tail = self.messages[-6:]
        middle = self.messages[1:-6]

        # Summarize middle messages
        summary_parts = []
        for msg in middle:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Truncate each message for the summary
                snippet = content[:200] + "..." if len(content) > 200 else content
                summary_parts.append(f"[{role}]: {snippet}")

        summary = "[Earlier conversation summary]\n" + "\n".join(summary_parts)
        summary_msg = {"role": "user", "content": summary}

        self.messages = [system, summary_msg] + tail

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
        """Make a non-streaming API call to get the model's plan. Returns plan text or None."""
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

            # If no tool calls, we're done
            if not tool_calls:
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

                # Ensure tool_args is a dict
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {"raw": tool_args}

                yield StreamChunk(
                    type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                )

                tool = self.tools.get(tool_name)
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
