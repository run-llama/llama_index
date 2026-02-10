"""Tests for configuration module."""

import os
from unittest.mock import patch

from llama_index.tools.mcp_code_execution.config import Config


class TestConfig:
    def test_default_values(self) -> None:
        """Test that defaults are sensible."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.executable == "/bin/bash"
            assert config.init_commands == []
            assert config.first_output_timeout == 30.0
            assert config.between_output_timeout == 15.0
            assert config.dialog_timeout == 5.0
            assert config.max_exec_timeout == 180.0
            assert config.log_dir == ""

    def test_env_override_executable(self) -> None:
        with patch.dict(os.environ, {"CODE_EXEC_EXECUTABLE": "/bin/zsh"}):
            config = Config()
            assert config.executable == "/bin/zsh"

    def test_env_override_init_commands(self) -> None:
        with patch.dict(
            os.environ,
            {"CODE_EXEC_INIT_COMMANDS": "source /venv/bin/activate;export FOO=bar"},
        ):
            config = Config()
            assert config.init_commands == [
                "source /venv/bin/activate",
                "export FOO=bar",
            ]

    def test_env_override_timeouts(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CODE_EXEC_FIRST_OUTPUT_TIMEOUT": "60",
                "CODE_EXEC_BETWEEN_OUTPUT_TIMEOUT": "30",
                "CODE_EXEC_DIALOG_TIMEOUT": "10",
                "CODE_EXEC_MAX_EXEC_TIMEOUT": "300",
            },
        ):
            config = Config()
            assert config.first_output_timeout == 60.0
            assert config.between_output_timeout == 30.0
            assert config.dialog_timeout == 10.0
            assert config.max_exec_timeout == 300.0

    def test_env_override_log_dir(self) -> None:
        with patch.dict(os.environ, {"CODE_EXEC_LOG_DIR": "/tmp/logs"}):
            config = Config()
            assert config.log_dir == "/tmp/logs"

    def test_empty_init_commands(self) -> None:
        with patch.dict(os.environ, {"CODE_EXEC_INIT_COMMANDS": ""}):
            config = Config()
            assert config.init_commands == []

    def test_init_commands_with_trailing_semicolons(self) -> None:
        with patch.dict(os.environ, {"CODE_EXEC_INIT_COMMANDS": "cmd1;;cmd2;"}):
            config = Config()
            assert config.init_commands == ["cmd1", "cmd2"]
