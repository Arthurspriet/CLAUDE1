"""MCP (Model Context Protocol) client manager.

Connects to MCP servers (stdio and SSE transports), discovers their tools,
and wraps each remote tool as a native BaseTool so it plugs into the
existing ToolRegistry seamlessly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude1.config import DATA_DIR
from claude1.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Guarded MCP import
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Guarded SSE import
try:
    from mcp.client.sse import sse_client
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False


# ── Config dataclass ─────────────────────────────────────────────────────────

@dataclass
class MCPServerConfig:
    """Parsed configuration for a single MCP server."""

    name: str
    transport: str = "stdio"  # "stdio" or "sse"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)


# ── Tool bridge ──────────────────────────────────────────────────────────────

class MCPToolBridge(BaseTool):
    """Wraps a single MCP tool as a native BaseTool.

    Bridges the synchronous ``execute()`` call to the async MCP
    ``session.call_tool()`` via the manager's event loop.
    """

    def __init__(
        self,
        working_dir: str,
        server_name: str,
        tool_info: dict,
        session: Any,
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__(working_dir)
        self._server_name = server_name
        self._tool_info = tool_info
        self._session = session
        self._loop = loop

    @property
    def name(self) -> str:
        raw_name = self._tool_info.get("name", "unknown")
        return f"mcp_{self._server_name}_{raw_name}"

    @property
    def description(self) -> str:
        desc = self._tool_info.get("description", "")
        return f"[MCP:{self._server_name}] {desc}"

    @property
    def parameters(self) -> dict:
        schema = self._tool_info.get("inputSchema", {})
        if schema:
            return schema
        return {"type": "object", "properties": {}}

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        if not MCP_AVAILABLE:
            return "Error: mcp package is not installed. Run: pip install mcp"

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._call_tool(kwargs), self._loop
            )
            result = future.result(timeout=120)
            return result
        except TimeoutError:
            return "Error: MCP tool call timed out after 120s"
        except Exception as e:
            return f"Error calling MCP tool: {e}"

    async def _call_tool(self, arguments: dict) -> str:
        result = await self._session.call_tool(
            self._tool_info["name"], arguments=arguments
        )
        # MCP returns content as a list of content blocks
        if hasattr(result, "content") and result.content:
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return str(result)


# ── Client manager ───────────────────────────────────────────────────────────

class MCPClientManager:
    """Loads MCP config, connects to servers, and provides tools."""

    GLOBAL_CONFIG = DATA_DIR / "mcp.json"
    PROJECT_CONFIG_NAME = ".claude1/mcp.json"

    def __init__(self, working_dir: str):
        self._working_dir = working_dir
        self._configs: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, Any] = {}
        self._tools: list[BaseTool] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        # Track async context managers for cleanup
        self._transports: dict[str, Any] = {}
        self._session_cms: dict[str, Any] = {}

    def load_config(self) -> dict[str, MCPServerConfig]:
        """Load and merge global + project MCP configs.

        Returns the merged dict of server configs.
        """
        merged: dict[str, Any] = {}

        # Global config
        if self.GLOBAL_CONFIG.exists():
            try:
                data = json.loads(self.GLOBAL_CONFIG.read_text())
                merged.update(data.get("mcpServers", {}))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read global MCP config: %s", e)

        # Project config (overrides global)
        project_cfg = Path(self._working_dir) / self.PROJECT_CONFIG_NAME
        if project_cfg.exists():
            try:
                data = json.loads(project_cfg.read_text())
                merged.update(data.get("mcpServers", {}))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read project MCP config: %s", e)

        # Parse into dataclass
        self._configs = {}
        for name, raw in merged.items():
            transport = raw.get("type", "stdio")
            self._configs[name] = MCPServerConfig(
                name=name,
                transport=transport,
                command=raw.get("command", ""),
                args=raw.get("args", []),
                env=raw.get("env", {}),
                url=raw.get("url", ""),
                headers=raw.get("headers", {}),
            )

        return self._configs

    def connect_all(self) -> list[BaseTool]:
        """Connect to all configured MCP servers and discover tools.

        Returns the list of MCPToolBridge instances.
        """
        if not MCP_AVAILABLE:
            logger.info("MCP package not installed — skipping MCP connections")
            return []

        if not self._configs:
            return []

        # Start a background event loop for async MCP operations
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="mcp-event-loop"
        )
        self._loop_thread.start()

        self._tools = []

        for name, cfg in self._configs.items():
            try:
                tools = self._connect_server(name, cfg)
                self._tools.extend(tools)
                logger.info("MCP server '%s': connected, %d tools", name, len(tools))
            except Exception as e:
                logger.warning("Failed to connect MCP server '%s': %s", name, e)

        return list(self._tools)

    def _connect_server(self, name: str, cfg: MCPServerConfig) -> list[MCPToolBridge]:
        """Connect to a single MCP server and return its tools."""
        assert self._loop is not None

        future = asyncio.run_coroutine_threadsafe(
            self._async_connect(name, cfg), self._loop
        )
        return future.result(timeout=30)

    async def _async_connect(
        self, name: str, cfg: MCPServerConfig
    ) -> list[MCPToolBridge]:
        """Async connection to a single MCP server."""
        if cfg.transport == "stdio":
            return await self._connect_stdio(name, cfg)
        elif cfg.transport == "sse":
            return await self._connect_sse(name, cfg)
        else:
            raise ValueError(f"Unknown MCP transport: {cfg.transport}")

    async def _connect_stdio(
        self, name: str, cfg: MCPServerConfig
    ) -> list[MCPToolBridge]:
        """Connect to a stdio MCP server."""
        env = {**os.environ, **cfg.env}
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=env,
        )

        # Enter the transport context manager
        transport_cm = stdio_client(params)
        read_stream, write_stream = await transport_cm.__aenter__()
        self._transports[name] = transport_cm

        # Enter the session context manager
        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        self._session_cms[name] = session_cm
        self._sessions[name] = session

        await session.initialize()

        return await self._discover_tools(name, session)

    async def _connect_sse(
        self, name: str, cfg: MCPServerConfig
    ) -> list[MCPToolBridge]:
        """Connect to an SSE MCP server."""
        if not SSE_AVAILABLE:
            raise RuntimeError("SSE client not available in installed mcp package")

        transport_cm = sse_client(cfg.url, headers=cfg.headers)
        read_stream, write_stream = await transport_cm.__aenter__()
        self._transports[name] = transport_cm

        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        self._session_cms[name] = session_cm
        self._sessions[name] = session

        await session.initialize()

        return await self._discover_tools(name, session)

    async def _discover_tools(
        self, server_name: str, session: Any
    ) -> list[MCPToolBridge]:
        """List tools from a connected session and wrap them as MCPToolBridge."""
        assert self._loop is not None

        result = await session.list_tools()
        tools_list = result.tools if hasattr(result, "tools") else []

        bridges = []
        for tool in tools_list:
            tool_info = {
                "name": getattr(tool, "name", "unknown"),
                "description": getattr(tool, "description", ""),
                "inputSchema": getattr(tool, "inputSchema", {}),
            }
            bridge = MCPToolBridge(
                working_dir=self._working_dir,
                server_name=server_name,
                tool_info=tool_info,
                session=session,
                loop=self._loop,
            )
            bridges.append(bridge)

        return bridges

    def get_tools(self) -> list[BaseTool]:
        """Return all discovered MCP tools."""
        return list(self._tools)

    def get_status(self) -> dict[str, Any]:
        """Return status information about connected servers."""
        status: dict[str, Any] = {
            "mcp_available": MCP_AVAILABLE,
            "servers": {},
            "total_tools": len(self._tools),
        }

        for name, cfg in self._configs.items():
            connected = name in self._sessions
            server_tools = [
                t for t in self._tools
                if isinstance(t, MCPToolBridge) and t._server_name == name
            ]
            status["servers"][name] = {
                "transport": cfg.transport,
                "connected": connected,
                "tools": len(server_tools),
                "command": cfg.command if cfg.transport == "stdio" else "",
                "url": cfg.url if cfg.transport == "sse" else "",
            }

        return status

    def reconnect_all(self) -> list[BaseTool]:
        """Disconnect all servers and reconnect."""
        self.close()
        self.load_config()
        return self.connect_all()

    def close(self):
        """Clean up all connections and stop the event loop."""
        if self._loop is not None and self._loop.is_running():
            # Close sessions and transports
            future = asyncio.run_coroutine_threadsafe(
                self._async_cleanup(), self._loop
            )
            try:
                future.result(timeout=10)
            except Exception:
                pass

            # Stop the event loop
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)

        self._sessions.clear()
        self._transports.clear()
        self._session_cms.clear()
        self._tools.clear()
        self._loop = None
        self._loop_thread = None

    async def _async_cleanup(self):
        """Async cleanup of sessions and transports."""
        for name, session_cm in list(self._session_cms.items()):
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass

        for name, transport_cm in list(self._transports.items()):
            try:
                await transport_cm.__aexit__(None, None, None)
            except Exception:
                pass

    def save_config(self, server_name: str, server_config: dict, global_scope: bool = True):
        """Save a server config to the config file.

        Args:
            server_name: Name for the server.
            server_config: Raw config dict for the server.
            global_scope: If True, save to global config; otherwise project.
        """
        if global_scope:
            config_path = self.GLOBAL_CONFIG
        else:
            config_path = Path(self._working_dir) / self.PROJECT_CONFIG_NAME

        config_path.parent.mkdir(parents=True, exist_ok=True)

        existing: dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        servers = existing.setdefault("mcpServers", {})
        servers[server_name] = server_config

        config_path.write_text(json.dumps(existing, indent=2) + "\n")
