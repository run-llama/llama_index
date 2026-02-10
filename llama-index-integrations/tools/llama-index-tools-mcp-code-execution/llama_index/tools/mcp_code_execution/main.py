"""Code Execution MCP Server.

A Model Context Protocol server that exposes code execution capabilities
for terminal commands and Python code. Uses FastMCP for protocol handling
and persistent TTY sessions for state management.

Usage:
    # As a console script (after installation):
    code-execution-mcp

    # Or directly:
    python -m llama_index.tools.mcp_code_execution.main
"""

from mcp.server.fastmcp import FastMCP

from .config import Config
from .print_style import log_info, setup_logging
from .tty_session import SessionManager

# Initialize configuration from environment variables
config = Config()

# Set up file logging if configured
setup_logging(config.log_dir)

# Initialize session manager
session_manager = SessionManager(config)

# Create MCP server
mcp = FastMCP("code-execution")

log_info("Code Execution MCP Server initialized")


@mcp.tool()
def execute_terminal(command: str, session: int = 0) -> str:
    """Execute a terminal command in the specified session.

    Run shell commands with full session persistence. Each session
    maintains its own shell state (environment variables, working
    directory, etc.).

    Args:
        command: The shell command to execute.
        session: Session (terminal window) number. Default is 0.
            Different session numbers maintain separate execution contexts.

    Returns:
        The accumulated terminal output from the command execution.
    """
    sess = session_manager.get_session(session)
    return sess.execute(command)


@mcp.tool()
def execute_python(code: str, session: int = 0) -> str:
    """Execute Python code via IPython in the specified session.

    Run Python code in an IPython session. IPython is automatically
    started on first use. The session maintains state between calls
    (variables, imports, etc.).

    Args:
        code: The Python code to execute.
        session: Session (terminal window) number. Default is 0.
            Different session numbers maintain separate IPython instances.

    Returns:
        The accumulated IPython output from the code execution.
    """
    sess = session_manager.get_session(session)
    return sess.execute_python(code)


@mcp.tool()
def get_output(session: int = 0) -> str:
    """Get accumulated output from a terminal session.

    Retrieve the full output buffer from a session, including any
    new output that has appeared since the last read.

    Args:
        session: Session (terminal window) number. Default is 0.

    Returns:
        The accumulated terminal output from the session.
    """
    sess = session_manager.get_session(session)
    return sess.get_output()


@mcp.tool()
def reset_terminal(session: int = 0, reason: str = "") -> str:
    """Reset a terminal session, closing and reopening it.

    This destroys the current shell process and starts a fresh one.
    All state (environment variables, working directory, running
    processes) will be lost.

    Args:
        session: Session (terminal window) number. Default is 0.
        reason: Optional reason for the reset, logged for debugging.

    Returns:
        Text confirmation of the reset.
    """
    return session_manager.reset_session(session, reason)


def main() -> None:
    """Entry point for the MCP server."""
    log_info("Starting Code Execution MCP Server via stdio")
    mcp.run("stdio")


if __name__ == "__main__":
    main()
