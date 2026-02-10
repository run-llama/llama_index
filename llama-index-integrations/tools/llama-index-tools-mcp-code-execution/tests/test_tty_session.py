"""Tests for TTY session management."""

import platform
import sys

import pytest

from llama_index.tools.mcp_code_execution.config import Config
from llama_index.tools.mcp_code_execution.tty_session import (
    SessionManager,
    TTYSession,
)


@pytest.fixture
def config() -> Config:
    """Create a config with short timeouts for testing."""
    return Config(
        executable="/bin/bash",
        init_commands=[],
        first_output_timeout=5.0,
        between_output_timeout=3.0,
        dialog_timeout=2.0,
        max_exec_timeout=10.0,
        log_dir="",
    )


@pytest.mark.skipif(
    platform.system() == "Windows", reason="TTY tests require Unix"
)
class TestTTYSession:
    def test_session_starts(self, config: Config) -> None:
        session = TTYSession(0, config)
        assert session.process is not None
        assert session.process.isalive()
        session._cleanup()

    def test_execute_simple_command(self, config: Config) -> None:
        session = TTYSession(0, config)
        output = session.execute("echo hello")
        assert "hello" in output
        session._cleanup()

    def test_execute_preserves_state(self, config: Config) -> None:
        session = TTYSession(0, config)
        session.execute("export TEST_VAR=12345")
        output = session.execute("echo $TEST_VAR")
        assert "12345" in output
        session._cleanup()

    def test_execute_working_directory(self, config: Config) -> None:
        session = TTYSession(0, config)
        session.execute("cd /tmp")
        output = session.execute("pwd")
        assert "/tmp" in output
        session._cleanup()

    def test_get_output(self, config: Config) -> None:
        session = TTYSession(0, config)
        session.execute("echo test_output")
        output = session.get_output()
        assert "test_output" in output
        session._cleanup()

    def test_reset(self, config: Config) -> None:
        session = TTYSession(0, config)
        session.execute("export RESET_VAR=abc")
        result = session.reset("testing")
        assert "reset" in result.lower()
        output = session.execute("echo $RESET_VAR")
        # After reset, the variable should not be set
        assert "abc" not in output or output.strip() == ""
        session._cleanup()

    def test_output_buffer_accumulates(self, config: Config) -> None:
        session = TTYSession(0, config)
        session.execute("echo line1")
        session.execute("echo line2")
        output = session.get_output()
        assert "line1" in output
        assert "line2" in output
        session._cleanup()


@pytest.mark.skipif(
    platform.system() == "Windows", reason="TTY tests require Unix"
)
class TestSessionManager:
    def test_creates_session_on_demand(self, config: Config) -> None:
        manager = SessionManager(config)
        session = manager.get_session(0)
        assert session is not None
        assert session.session_id == 0
        manager.cleanup_all()

    def test_returns_same_session(self, config: Config) -> None:
        manager = SessionManager(config)
        session1 = manager.get_session(0)
        session2 = manager.get_session(0)
        assert session1 is session2
        manager.cleanup_all()

    def test_different_sessions(self, config: Config) -> None:
        manager = SessionManager(config)
        session0 = manager.get_session(0)
        session1 = manager.get_session(1)
        assert session0 is not session1
        assert session0.session_id == 0
        assert session1.session_id == 1
        manager.cleanup_all()

    def test_session_isolation(self, config: Config) -> None:
        manager = SessionManager(config)
        session0 = manager.get_session(0)
        session1 = manager.get_session(1)
        session0.execute("export ISOLATED_VAR=from_session0")
        output = session1.execute("echo $ISOLATED_VAR")
        # Variable should not leak between sessions
        assert "from_session0" not in output or output.strip() == ""
        manager.cleanup_all()

    def test_reset_session(self, config: Config) -> None:
        manager = SessionManager(config)
        manager.get_session(0)
        result = manager.reset_session(0, "test reset")
        assert "reset" in result.lower()
        manager.cleanup_all()

    def test_reset_nonexistent_session(self, config: Config) -> None:
        manager = SessionManager(config)
        result = manager.reset_session(99, "new")
        assert "99" in result
        manager.cleanup_all()

    def test_cleanup_all(self, config: Config) -> None:
        manager = SessionManager(config)
        manager.get_session(0)
        manager.get_session(1)
        manager.cleanup_all()
        assert len(manager.sessions) == 0
