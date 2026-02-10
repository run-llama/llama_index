"""Configuration for the Code Execution MCP Server.

All settings are loaded from environment variables with sensible defaults.
"""

import os
import platform
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration loaded from environment variables.

    Environment Variables:
        CODE_EXEC_EXECUTABLE: Shell executable path
            (default: /bin/bash on Unix, powershell.exe on Windows)
        CODE_EXEC_INIT_COMMANDS: Semicolon-separated commands to run on session start
        CODE_EXEC_FIRST_OUTPUT_TIMEOUT: Seconds to wait for first output (default: 30)
        CODE_EXEC_BETWEEN_OUTPUT_TIMEOUT: Seconds between output chunks (default: 15)
        CODE_EXEC_DIALOG_TIMEOUT: Seconds to detect dialog prompts (default: 5)
        CODE_EXEC_MAX_EXEC_TIMEOUT: Maximum total execution time (default: 180)
        CODE_EXEC_LOG_DIR: Directory for log files (empty = disabled)
    """

    executable: str = field(
        default_factory=lambda: os.environ.get(
            "CODE_EXEC_EXECUTABLE",
            "/bin/bash" if platform.system() != "Windows" else "powershell.exe",
        )
    )

    init_commands: list[str] = field(
        default_factory=lambda: [
            cmd.strip()
            for cmd in os.environ.get("CODE_EXEC_INIT_COMMANDS", "").split(";")
            if cmd.strip()
        ]
    )

    first_output_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("CODE_EXEC_FIRST_OUTPUT_TIMEOUT", "30")
        )
    )

    between_output_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("CODE_EXEC_BETWEEN_OUTPUT_TIMEOUT", "15")
        )
    )

    dialog_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("CODE_EXEC_DIALOG_TIMEOUT", "5")
        )
    )

    max_exec_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("CODE_EXEC_MAX_EXEC_TIMEOUT", "180")
        )
    )

    log_dir: str = field(
        default_factory=lambda: os.environ.get("CODE_EXEC_LOG_DIR", "")
    )
