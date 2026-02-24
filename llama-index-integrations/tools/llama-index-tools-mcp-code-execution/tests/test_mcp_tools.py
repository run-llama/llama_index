"""Integration tests for MCP tool functions.

These tests verify the actual MCP tool functions (execute_terminal,
execute_python, get_output, reset_terminal) work correctly end-to-end.
"""

import platform
import os
from unittest.mock import patch

import pytest

# Avoid importing the module-level mcp server which starts session manager eagerly.
# Instead, test tools through the session manager directly and verify registration.


@pytest.mark.skipif(
    platform.system() == "Windows", reason="TTY tests require Unix"
)
class TestMCPToolRegistration:
    def test_all_tools_registered(self) -> None:
        from llama_index.tools.mcp_code_execution import mcp

        tool_names = list(mcp._tool_manager._tools.keys())
        assert "execute_terminal" in tool_names
        assert "execute_python" in tool_names
        assert "get_output" in tool_names
        assert "reset_terminal" in tool_names

    def test_server_name(self) -> None:
        from llama_index.tools.mcp_code_execution import mcp

        assert mcp.name == "code-execution"


@pytest.mark.skipif(
    platform.system() == "Windows", reason="TTY tests require Unix"
)
class TestExecuteTerminalTool:
    def test_basic_echo(self) -> None:
        from llama_index.tools.mcp_code_execution.main import (
            execute_terminal,
            session_manager,
        )

        result = execute_terminal("echo mcp_tool_test")
        assert "mcp_tool_test" in result
        session_manager.cleanup_all()

    def test_piped_commands(self) -> None:
        from llama_index.tools.mcp_code_execution.main import (
            execute_terminal,
            session_manager,
        )

        result = execute_terminal("echo 'abc def ghi' | wc -w")
        assert "3" in result
        session_manager.cleanup_all()

    def test_multiline_output(self) -> None:
        from llama_index.tools.mcp_code_execution.main import (
            execute_terminal,
            session_manager,
        )

        result = execute_terminal("echo -e 'line1\\nline2\\nline3'")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        session_manager.cleanup_all()

    def test_exit_code_in_marker(self) -> None:
        """Commands with non-zero exit should still return output."""
        from llama_index.tools.mcp_code_execution.main import (
            execute_terminal,
            session_manager,
        )

        result = execute_terminal("ls /nonexistent_path_xyz 2>&1")
        assert "No such file" in result or "cannot access" in result
        session_manager.cleanup_all()

    def test_session_parameter(self) -> None:
        from llama_index.tools.mcp_code_execution.main import (
            execute_terminal,
            session_manager,
        )

        execute_terminal("export SESSION_TEST=42", session=5)
        result = execute_terminal("echo $SESSION_TEST", session=5)
        assert "42" in result
        session_manager.cleanup_all()


@pytest.mark.skipif(
    platform.system() == "Windows", reason="TTY tests require Unix"
)
class TestGetOutputTool:
    def test_get_output_returns_buffer(self) -> None:
        from llama_index.tools.mcp_code_execution.main import (
            execute_terminal,
            get_output,
            session_manager,
        )

        execute_terminal("echo buffer_content", session=10)
        result = get_output(session=10)
        assert "buffer_content" in result
        session_manager.cleanup_all()


@pytest.mark.skipif(
    platform.system() == "Windows", reason="TTY tests require Unix"
)
class TestResetTerminalTool:
    def test_reset_clears_state(self) -> None:
        from llama_index.tools.mcp_code_execution.main import (
            execute_terminal,
            reset_terminal,
            session_manager,
        )

        execute_terminal("export RESET_ME=yes", session=20)
        result = reset_terminal(session=20, reason="integration test")
        assert "reset" in result.lower()

        out = execute_terminal("echo $RESET_ME", session=20)
        assert "yes" not in out or out.strip() == ""
        session_manager.cleanup_all()

    def test_reset_with_reason(self) -> None:
        from llama_index.tools.mcp_code_execution.main import (
            reset_terminal,
            session_manager,
        )

        result = reset_terminal(session=30, reason="custom reason")
        assert "custom reason" in result
        session_manager.cleanup_all()
