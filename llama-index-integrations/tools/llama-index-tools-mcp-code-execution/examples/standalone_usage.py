"""
Example: Using the Code Execution MCP Server directly (no LLM required).

This example demonstrates the server's core capabilities by calling the
session manager directly -- useful for testing or scripting.

Prerequisites:
    pip install llama-index-tools-mcp-code-execution

Usage:
    python standalone_usage.py
"""

from llama_index.tools.mcp_code_execution.config import Config
from llama_index.tools.mcp_code_execution.tty_session import SessionManager


def main() -> None:
    # Create a config with custom timeouts
    config = Config(
        executable="/bin/bash",
        init_commands=[],
        first_output_timeout=10,
        between_output_timeout=5,
        max_exec_timeout=30,
    )

    manager = SessionManager(config)

    # --- Terminal commands ---
    session = manager.get_session(0)

    print("=== Execute terminal commands ===")
    out = session.execute("echo 'Hello from session 0!'")
    print(f"Output: {out}\n")

    out = session.execute("date && hostname")
    print(f"Date/host: {out}\n")

    # State persists between commands
    session.execute("export MY_VAR='persistent'")
    out = session.execute("echo $MY_VAR")
    print(f"Persistent var: {out}\n")

    # --- Multiple sessions ---
    print("=== Session isolation ===")
    s1 = manager.get_session(1)
    s1.execute("cd /tmp && export SESSION_ID=one")
    s0_check = session.execute("echo session0_dir=$(pwd) session0_id=$SESSION_ID")
    s1_check = s1.execute("echo session1_dir=$(pwd) session1_id=$SESSION_ID")
    print(f"Session 0: {s0_check}")
    print(f"Session 1: {s1_check}\n")

    # --- Get accumulated output ---
    print("=== Output buffer ===")
    buf = session.get_output()
    print(f"Session 0 buffer length: {len(buf)} chars\n")

    # --- Reset a session ---
    print("=== Reset session ===")
    result = manager.reset_session(1, reason="demo cleanup")
    print(f"Reset result: {result}\n")

    # Clean up
    manager.cleanup_all()
    print("Done!")


if __name__ == "__main__":
    main()
