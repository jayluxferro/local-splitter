"""Transport layer: MCP stdio server and OpenAI-compatible HTTP proxy.

Both transports call the same pipeline orchestrator.
"""

from .http_proxy import create_app
from .mcp_server import create_mcp_server

__all__ = ["create_app", "create_mcp_server"]
