#!/usr/bin/env python3
"""Entrypoint script to run Gaggimate MCP server over HTTP/SSE or stdio."""

import os
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from gaggimate_mcp.server import mcp

if __name__ == "__main__":
    host = os.getenv("FASTMCP_HOST", os.getenv("MCP_HOST", "0.0.0.0"))
    port = int(os.getenv("FASTMCP_PORT", os.getenv("MCP_PORT", os.getenv("PORT", "3000"))))
    transport = os.getenv("MCP_TRANSPORT", "sse")

    try:
        from mcp.server.transport_security import TransportSecuritySettings
        security_settings = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
            allowed_hosts=["*"],
            allowed_origins=["*"],
        )
    except ImportError:
        security_settings = None

    if hasattr(mcp, "settings"):
        mcp.settings.host = host
        mcp.settings.port = port
        if security_settings is not None and hasattr(mcp.settings, "transport_security"):
            mcp.settings.transport_security = security_settings

    print(f"Starting Gaggimate MCP server on {host}:{port} using {transport} transport (DNS rebinding protection disabled)...")
    mcp.run(transport=transport)
