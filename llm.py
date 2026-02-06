"""LLM interface with agentic tool-calling loop."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import ollama

from config import AppConfig, MAX_TOOL_ITERATIONS, MAX_RETRIES, RETRY_BASE_DELAY, CONTEXT_WINDOW_RESERVE
from system_prompt import build_system_prompt
from tools import ToolRegistry


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

        # Initialize system prompt
        self._set_system_prompt()

    def _set_system_prompt(self):
        """Set or update the system prompt."""
        prompt = build_system_prompt(self.config.working_dir, self.config.model, compact=self.config.compact)
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": prompt}
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def update_model(self, model: str):
        """Switch to a different model."""
        self.config.model = model
        self._set_system_prompt()

    def clear_history(self):
        """Clear conversation history, keeping only the system prompt."""
        self.messages = [self.messages[0]] if self.messages else []
        self._set_system_prompt()

    def cancel(self):
        """Signal cancellation of current generation."""
        self._cancelled = True

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

    def send_message(self, user_input: str) -> Iterator[StreamChunk]:
        """Send a user message and yield streaming chunks through the agentic loop."""
        self._cancelled = False
        self.messages.append({"role": "user", "content": user_input})

        # Truncate if context is getting large
        self._truncate_if_needed()

        tool_defs = self.tools.ollama_tool_definitions()

        for iteration in range(MAX_TOOL_ITERATIONS):
            if self._cancelled:
                yield StreamChunk(type="done", content="[cancelled]")
                return

            # Debug info before API call
            if self.config.verbose:
                yield StreamChunk(
                    type="debug",
                    content=f"API call: {len(self.messages)} messages, model={self.config.model}, tools={len(tool_defs)}"
                )

            # Emit thinking start
            yield StreamChunk(type="thinking_start")

            # Build options dict
            options: dict[str, Any] = {"num_ctx": self.config.num_ctx}
            if self.config.temperature is not None:
                options["temperature"] = self.config.temperature

            # Retry loop with exponential backoff
            stream = None
            for attempt in range(MAX_RETRIES):
                try:
                    stream = ollama.chat(
                        model=self.config.model,
                        messages=self.messages,
                        tools=tool_defs,
                        stream=True,
                        options=options,
                    )
                    break
                except (ollama.ResponseError, ConnectionError, TimeoutError) as e:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        yield StreamChunk(type="text", content=f"\n[Retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s...]\n")
                        time.sleep(delay)
                    else:
                        yield StreamChunk(type="thinking_stop")
                        yield StreamChunk(type="error", content=f"Failed after {MAX_RETRIES} retries: {e}")
                        return
                except Exception as e:
                    yield StreamChunk(type="thinking_stop")
                    yield StreamChunk(type="error", content=f"Error: {e}")
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
                for chunk in stream:
                    if self._cancelled:
                        if not thinking_stopped:
                            yield StreamChunk(type="thinking_stop")
                        yield StreamChunk(type="done", content="[cancelled]")
                        return

                    msg = chunk.get("message", {})

                    # Stream text content
                    token = msg.get("content", "")
                    if token:
                        if not thinking_stopped:
                            yield StreamChunk(type="thinking_stop")
                            thinking_stopped = True
                        full_text += token
                        yield StreamChunk(type="text", content=token)

                    # Collect tool calls
                    if msg.get("tool_calls"):
                        if not thinking_stopped:
                            yield StreamChunk(type="thinking_stop")
                            thinking_stopped = True
                        tool_calls.extend(msg["tool_calls"])

                    # Capture stats from the final chunk
                    if chunk.get("done"):
                        prompt_eval_count = chunk.get("prompt_eval_count", 0)
                        eval_count = chunk.get("eval_count", 0)
                        eval_duration = chunk.get("eval_duration", 0)
                        self._last_prompt_eval_count = prompt_eval_count

                        if prompt_eval_count or eval_count:
                            stats_data = {
                                "prompt_eval_count": prompt_eval_count,
                                "eval_count": eval_count,
                                "eval_duration": eval_duration,
                            }
                            yield StreamChunk(type="stats", content=json.dumps(stats_data))

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

            # Execute tool calls
            for tc in tool_calls:
                if self._cancelled:
                    yield StreamChunk(type="done", content="[cancelled]")
                    return

                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                tool_args = func.get("arguments", {})

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
                self.messages.append({
                    "role": "tool",
                    "content": result,
                })

            # Loop back for the model to process tool results

        # Exceeded max iterations
        yield StreamChunk(
            type="error",
            content=f"Reached maximum tool iterations ({MAX_TOOL_ITERATIONS}). Stopping.",
        )
