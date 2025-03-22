import os
from unittest.mock import MagicMock, patch

import pytest
from e2b_code_interpreter.models import Execution, Logs

from llama_index.server.tools.interpreter import E2BCodeInterpreter


class TestE2BCodeInterpreter:
    @pytest.fixture()
    def sandbox(self):  # type: ignore
        """Create a mock Sandbox with no API key requirement."""
        mock_sandbox = MagicMock()
        mock_sandbox.files = MagicMock()
        mock_sandbox.files.write = MagicMock()
        mock_sandbox.run_code = MagicMock()
        return mock_sandbox

    @pytest.fixture()
    def code_interpreter(self, sandbox):  # type: ignore
        """Create E2BCodeInterpreter that uses the mock Sandbox."""
        with patch.dict(os.environ, {"E2B_API_KEY": "dummy_key"}):
            interpreter = E2BCodeInterpreter()
            interpreter.interpreter = sandbox
            return interpreter

    def test_interpret_success(self, code_interpreter, sandbox) -> None:  # type: ignore
        """Test successful code execution."""
        # Mock execution result
        mock_execution = Execution()
        mock_execution.error = None
        mock_execution.results = []
        mock_execution.logs = Logs(
            stdout="stdout", stderr="", display_data="", error=""
        )
        sandbox.run_code.return_value = mock_execution

        # Run the code
        result = code_interpreter.interpret("print('hello')")

        # Verify
        sandbox.run_code.assert_called_once_with("print('hello')")
        assert result.is_error is False
        assert result.logs == mock_execution.logs

    def test_interpret_error(self, code_interpreter, sandbox) -> None:  # type: ignore
        """Test error in code execution."""
        # Mock execution result with error
        mock_execution = Execution()
        mock_execution.error = "Test error"
        mock_execution.logs = Logs(
            stdout="", stderr="error", display_data="", error="Test error"
        )
        sandbox.run_code.return_value = mock_execution

        # Run the code
        result = code_interpreter.interpret("bad code")

        # Verify
        assert result.is_error is True
        assert "Error: Test error" in result.error_message
        sandbox.kill.assert_called_once()

    def test_to_tool(self, code_interpreter) -> None:  # type: ignore
        """Test tool conversion."""
        tool = code_interpreter.to_tool()
        assert tool.fn == code_interpreter.interpret
