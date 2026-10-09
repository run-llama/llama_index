"""Code Interpreter tool spec."""

import logging
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

logger = logging.getLogger(__name__)

BLOCKED_SECRET_ENV_PREFIXES = (
    "OPENAI_",
    "ANTHROPIC_",
    "AWS_",
    "AZURE_",
    "COHERE_",
    "GOOGLE_",
    "HUGGINGFACE_",
    "HF_",
    "GITHUB_",
    "GH_",
    "VETTO_",
)

BLOCKED_SECRET_ENV_KEYS = {
    "SECRET",
    "API_KEY",
    "TOKEN",
    "PASSWORD",
    "PRIVATE_KEY",
    "ACCESS_KEY",
}


def _build_clean_sandboxed_env(
    custom_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a clean environment dictionary for sandboxed execution, stripping host secrets."""
    base_env: Dict[str, str] = {}
    for key, value in os.environ.items():
        key_upper = key.upper()
        if any(key_upper.startswith(prefix) for prefix in BLOCKED_SECRET_ENV_PREFIXES):
            continue
        if any(secret in key_upper for secret in BLOCKED_SECRET_ENV_KEYS):
            continue
        base_env[key] = value

    if custom_env:
        base_env.update(custom_env)

    return base_env


class CodeInterpreterToolSpec(BaseToolSpec):
    """
    Code Interpreter tool spec.

    WARNING: Without sandboxing, this tool executes arbitrary code on the host machine.
    To prevent malicious execution and host credential exposure, set sandbox="vetto"
    or instantiate VettoCodeInterpreterToolSpec.

    """

    spec_functions = ["code_interpreter"]

    def __init__(
        self,
        sandbox: Optional[str] = None,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        fail_closed: bool = True,
        net_mode: Optional[str] = "off",
    ) -> None:
        """Initialize the CodeInterpreterToolSpec.

        Args:
            sandbox: Optional sandboxing backend. Set to "vetto" for containerless kernel
                sandbox execution (<4ms cold start, Landlock LSM / macOS Seatbelt / Windows LPAC),
                or None for standard unconfined execution.
            working_dir: Optional workspace boundary for filesystem containment.
            timeout: Optional maximum execution wall-clock timeout in seconds.
            fail_closed: If True (default) and sandbox="vetto" is requested but the vetto
                CLI binary is not installed in PATH, raise RuntimeError instead of running unconfined.
            net_mode: Network policy for sandboxed execution ("off" for airgap, "host", or None).
        """
        self.sandbox = sandbox.lower().strip() if isinstance(sandbox, str) else None
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        self.timeout = timeout
        self.fail_closed = fail_closed
        self.net_mode = net_mode

    def code_interpreter(self, code: str) -> str:
        """
        A function to execute python code, and return the stdout and stderr.

        You should import any libraries that you wish to use. You have access to any libraries the user has installed.

        The code passed to this function is executed in isolation. It should be complete at the time it is passed to this function.

        You should interpret the output and errors returned from this function, and attempt to fix any problems.
        If you cannot fix the error, show the code to the user and ask for help

        It is not possible to return graphics or other complicated data from this function. If the user cannot see the output, save it to a file and tell the user.
        """
        if self.sandbox == "vetto":
            return self._run_sandboxed_vetto(code)

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, timeout=self.timeout
        )
        stdout = (
            result.stdout.decode("utf-8", errors="replace")
            if isinstance(result.stdout, bytes)
            else str(result.stdout)
        )
        stderr = (
            result.stderr.decode("utf-8", errors="replace")
            if isinstance(result.stderr, bytes)
            else str(result.stderr)
        )
        return f"StdOut:\n{stdout}\nStdErr:\n{stderr}"

    def _run_sandboxed_vetto(self, code: str) -> str:
        vetto_bin = shutil.which("vetto")
        if not vetto_bin:
            if self.fail_closed:
                raise RuntimeError(
                    "Vetto sandbox execution requested (sandbox='vetto'), but 'vetto' executable was not found in PATH. "
                    "Install Vetto (https://github.com/shleder/vetto) or set sandbox=None to allow unconfined host execution."
                )
            logger.warning(
                "Vetto binary not found in PATH; falling back to unconfined execution because fail_closed=False."
            )
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, timeout=self.timeout
            )
            stdout = (
                result.stdout.decode("utf-8", errors="replace")
                if isinstance(result.stdout, bytes)
                else str(result.stdout)
            )
            stderr = (
                result.stderr.decode("utf-8", errors="replace")
                if isinstance(result.stderr, bytes)
                else str(result.stderr)
            )
            return f"StdOut:\n{stdout}\nStdErr:\n{stderr}"

        cmd: List[str] = [vetto_bin, "run"]
        if self.working_dir:
            cmd.extend(["--workspace", self.working_dir])
        if self.timeout is not None:
            cmd.append(f"--timeout={self.timeout}s")
        if self.net_mode:
            cmd.append(f"--net={self.net_mode}")

        cmd.extend(["--", sys.executable, "-c", code])
        env = _build_clean_sandboxed_env()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                env=env,
                cwd=self.working_dir,
                timeout=self.timeout,
            )
            stdout = (
                result.stdout.decode("utf-8", errors="replace")
                if isinstance(result.stdout, bytes)
                else str(result.stdout)
            )
            stderr = (
                result.stderr.decode("utf-8", errors="replace")
                if isinstance(result.stderr, bytes)
                else str(result.stderr)
            )
            return f"StdOut:\n{stdout}\nStdErr:\n{stderr}"
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
            )
            stderr = (
                exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            )
            return (
                f"StdOut:\n{stdout}\nStdErr:\n"
                f"Command timed out after {self.timeout} seconds.\n{stderr}"
            )


class VettoCodeInterpreterToolSpec(CodeInterpreterToolSpec):
    """
    Vetto-sandboxed Code Interpreter tool spec.

    Executes Python code with containerless, sub-4ms cold start kernel-level isolation
    (Linux Landlock LSM ABI 1-6 / macOS Seatbelt / Windows LPAC), preventing host filesystem
    tampering, secret exfiltration, and socket abuse.
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = 30,
        fail_closed: bool = True,
        net_mode: Optional[str] = "off",
    ) -> None:
        super().__init__(
            sandbox="vetto",
            working_dir=working_dir,
            timeout=timeout,
            fail_closed=fail_closed,
            net_mode=net_mode,
        )
