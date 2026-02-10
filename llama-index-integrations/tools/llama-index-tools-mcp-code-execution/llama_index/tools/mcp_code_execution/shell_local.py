"""Local shell interface for spawning shell processes."""

import platform
import shutil

import pexpect


def spawn_shell(
    executable: str, rows: int = 50, cols: int = 200
) -> pexpect.spawn:
    """Spawn a new shell process using pexpect.

    Args:
        executable: Path or name of the shell executable.
        rows: Terminal row count.
        cols: Terminal column count.

    Returns:
        A pexpect.spawn child process.

    Raises:
        NotImplementedError: If running on Windows (use WSL instead).
        FileNotFoundError: If the executable cannot be found.
    """
    if platform.system() == "Windows":
        raise NotImplementedError(
            "Direct Windows support requires pywinpty. Use WSL or Linux instead."
        )

    exec_path = shutil.which(executable) or executable

    child = pexpect.spawn(
        exec_path,
        encoding="utf-8",
        timeout=30,
        dimensions=(rows, cols),
    )

    return child
