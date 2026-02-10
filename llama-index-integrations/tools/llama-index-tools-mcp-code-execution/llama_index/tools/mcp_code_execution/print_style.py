"""Output styling and logging utilities."""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("code-execution-mcp")


def setup_logging(log_dir: str) -> None:
    """Set up file logging if a log directory is specified.

    Args:
        log_dir: Path to the log directory. Empty string disables file logging.
    """
    if not log_dir:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"code_exec_{timestamp}.log"

    handler = logging.FileHandler(log_file)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def log_command(session: int, command: str) -> None:
    """Log a command execution."""
    logger.debug("[Session %d] Command: %s", session, command)


def log_output(session: int, output: str) -> None:
    """Log command output (truncated to 500 chars)."""
    logger.debug("[Session %d] Output: %s", session, output[:500])


def log_error(session: int, error: str) -> None:
    """Log an error."""
    logger.error("[Session %d] Error: %s", session, error)


def log_info(message: str) -> None:
    """Log an informational message."""
    logger.info(message)
