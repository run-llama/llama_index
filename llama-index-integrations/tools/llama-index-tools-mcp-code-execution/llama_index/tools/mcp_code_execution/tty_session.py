"""TTY session management for persistent shell and IPython sessions.

Provides session lifecycle management, smart output reading with timeout
handling, prompt detection, and dialog detection. Inspired by Agent Zero's
battle-tested code execution implementation.
"""

import re
import time
from typing import Optional

import pexpect

from .config import Config
from .print_style import log_command, log_error, log_info, log_output
from .shell_local import spawn_shell
from .strings import (
    clean_ansi,
    clean_carriage_returns,
    detect_dialog,
    detect_ipython_prompt,
    detect_prompt,
    strip_trailing_prompt,
    truncate_output,
)

# Unique marker for detecting command completion
_MARKER = "___CMD_DONE_MARKER___"
_MARKER_ECHO = f'echo "{_MARKER}$?"'


class TTYSession:
    """Manages a single TTY session with a shell process.

    Each session maintains a persistent shell process with full state
    (environment variables, working directory, etc.) preserved between
    command executions.
    """

    def __init__(self, session_id: int, config: Config) -> None:
        self.session_id = session_id
        self.config = config
        self.process: Optional[pexpect.spawn] = None
        self.output_buffer: str = ""
        self._ipython_started: bool = False
        self._start()

    def _start(self) -> None:
        """Start the shell process and run init commands."""
        try:
            self.process = spawn_shell(self.config.executable)
            # Wait for shell to initialize
            time.sleep(0.5)
            self._read_available()
            self.output_buffer = ""

            # Run init commands
            for cmd in self.config.init_commands:
                self._execute_raw(cmd)
                self.output_buffer = ""

            log_info(
                f"Session {self.session_id} started with {self.config.executable}"
            )
        except Exception as e:
            log_error(self.session_id, f"Failed to start session: {e}")
            raise

    def _read_available(self) -> str:
        """Read all currently available output from the process without blocking."""
        output = ""
        if self.process is None:
            return output

        try:
            while True:
                try:
                    chunk = self.process.read_nonblocking(size=4096, timeout=0.1)
                    output += chunk
                except pexpect.TIMEOUT:
                    break
                except pexpect.EOF:
                    break
        except Exception:
            pass

        return output

    def _execute_raw(
        self, command: str, timeout: Optional[float] = None
    ) -> str:
        """Execute a command using prompt-based completion detection.

        Used for init commands and internal operations where marker-based
        detection is not needed.
        """
        if self.process is None or not self.process.isalive():
            self._start()

        max_timeout = timeout or self.config.max_exec_timeout

        assert self.process is not None
        self.process.sendline(command)

        output = ""
        start_time = time.time()
        first_output_received = False
        last_output_time = start_time

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_timeout:
                break

            chunk = self._read_available()
            if chunk:
                output += chunk
                first_output_received = True
                last_output_time = time.time()
            else:
                if first_output_received:
                    since_last = time.time() - last_output_time
                    if since_last > self.config.between_output_timeout:
                        break
                    if detect_prompt(output):
                        break
                else:
                    if time.time() - start_time > self.config.first_output_timeout:
                        break
                time.sleep(0.05)

        self.output_buffer += output
        return output

    def execute(self, command: str) -> str:
        """Execute a terminal command and return the output.

        Uses a unique marker to reliably detect when the command has
        finished executing, even for commands that produce no output.

        Args:
            command: The shell command to execute.

        Returns:
            Cleaned output from the command execution.
        """
        if self.process is None or not self.process.isalive():
            self._start()

        log_command(self.session_id, command)

        # Chain the command with a marker echo for completion detection
        full_command = f"{command}; {_MARKER_ECHO}"

        assert self.process is not None
        self.process.sendline(full_command)

        output = ""
        start_time = time.time()
        first_output_received = False
        last_output_time = start_time

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.config.max_exec_timeout:
                output += "\n[MAX EXECUTION TIMEOUT REACHED]"
                break

            chunk = self._read_available()
            if chunk:
                output += chunk
                first_output_received = True
                last_output_time = time.time()

                # Check for completion marker
                if _MARKER in output:
                    break

                # Check for interactive dialog
                if detect_dialog(chunk):
                    output += (
                        "\n[DIALOG DETECTED - Interactive input may be required]"
                    )
                    break
            else:
                if first_output_received:
                    since_last = time.time() - last_output_time
                    if since_last > self.config.between_output_timeout:
                        break
                    if (
                        since_last > self.config.dialog_timeout
                        and detect_dialog(output)
                    ):
                        output += (
                            "\n[DIALOG DETECTED"
                            " - Interactive input may be required]"
                        )
                        break
                else:
                    if time.time() - start_time > self.config.first_output_timeout:
                        output += "\n[FIRST OUTPUT TIMEOUT]"
                        break
                time.sleep(0.05)

        cleaned = self._clean_terminal_output(output, command)
        self.output_buffer += cleaned

        log_output(self.session_id, cleaned)
        return truncate_output(cleaned)

    def execute_python(self, code: str) -> str:
        """Execute Python code via IPython.

        IPython is started automatically on first use. The session state
        (variables, imports, etc.) persists between calls.

        Args:
            code: Python code to execute.

        Returns:
            Cleaned output from the code execution.
        """
        if not self._ipython_started:
            self._start_ipython()

        log_command(self.session_id, f"[Python] {code}")

        if self.process is None or not self.process.isalive():
            self._start()
            self._start_ipython()

        assert self.process is not None

        # For multiline code, use %cpaste to avoid indentation issues
        lines = code.strip().split("\n")
        if len(lines) > 1:
            self.process.sendline("%cpaste -q")
            time.sleep(0.3)
            self._read_available()  # Clear cpaste prompt

            for line in lines:
                self.process.sendline(line)
                time.sleep(0.05)

            self.process.sendline("--")
        else:
            self.process.sendline(code)

        # Wait for output with IPython prompt detection
        output = ""
        start_time = time.time()
        first_output_received = False
        last_output_time = start_time

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.config.max_exec_timeout:
                output += "\n[MAX EXECUTION TIMEOUT REACHED]"
                break

            chunk = self._read_available()
            if chunk:
                output += chunk
                first_output_received = True
                last_output_time = time.time()

                # Check for IPython prompt indicating command is done
                if detect_ipython_prompt(output):
                    break
            else:
                if first_output_received:
                    since_last = time.time() - last_output_time
                    if since_last > self.config.between_output_timeout:
                        break
                else:
                    if time.time() - start_time > self.config.first_output_timeout:
                        output += "\n[FIRST OUTPUT TIMEOUT]"
                        break
                time.sleep(0.05)

        cleaned = self._clean_python_output(output)
        self.output_buffer += cleaned

        log_output(self.session_id, cleaned)
        return truncate_output(cleaned)

    def _start_ipython(self) -> None:
        """Start IPython in the current session."""
        self._execute_raw("ipython --no-banner --no-confirm-exit", timeout=15)
        self.output_buffer = ""
        self._ipython_started = True
        log_info(f"Session {self.session_id}: IPython started")

    def get_output(self) -> str:
        """Get accumulated output buffer, including any new output.

        Returns:
            The full output buffer from this session.
        """
        if self.process and self.process.isalive():
            new_output = self._read_available()
            if new_output:
                self.output_buffer += clean_ansi(new_output)
        return truncate_output(self.output_buffer)

    def reset(self, reason: str = "") -> str:
        """Reset the session by closing and reopening the shell.

        Args:
            reason: Optional reason for the reset (logged).

        Returns:
            Confirmation message.
        """
        log_info(f"Session {self.session_id} reset. Reason: {reason}")

        self._cleanup()
        self.output_buffer = ""
        self._ipython_started = False
        self._start()

        return (
            f"Session {self.session_id} has been reset."
            f" Reason: {reason or 'manual reset'}"
        )

    def _cleanup(self) -> None:
        """Clean up the shell process."""
        if self.process is not None:
            try:
                self.process.close(force=True)
            except Exception:
                pass
            self.process = None

    def _clean_terminal_output(self, output: str, command: str) -> str:
        """Clean raw terminal output by removing echoed command, markers,
        carriage returns, and trailing prompts."""
        cleaned = clean_ansi(output)
        cleaned = clean_carriage_returns(cleaned)

        lines = cleaned.split("\n")
        result_lines = []
        skip_first_match = True

        for line in lines:
            stripped = line.strip()
            # Skip the echoed command line
            if skip_first_match and command.strip() in stripped:
                skip_first_match = False
                continue
            # Skip marker-related lines
            if _MARKER in stripped:
                continue
            if f'echo "{_MARKER}' in stripped:
                continue
            result_lines.append(line)

        result = "\n".join(result_lines).strip()
        return strip_trailing_prompt(result)

    def _clean_python_output(self, output: str) -> str:
        """Clean IPython output by removing prompts, ANSI codes,
        and carriage returns."""
        cleaned = clean_ansi(output)
        cleaned = clean_carriage_returns(cleaned)
        # Remove trailing IPython prompt
        cleaned = re.sub(r"\s*In\s*\[\d+\]:\s*$", "", cleaned)
        return cleaned.strip()

    def __del__(self) -> None:
        self._cleanup()


class SessionManager:
    """Manages multiple TTY sessions indexed by session number."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.sessions: dict[int, TTYSession] = {}

    def get_session(self, session_id: int = 0) -> TTYSession:
        """Get an existing session or create a new one.

        Args:
            session_id: The session number (default: 0).

        Returns:
            The TTYSession for the given ID.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = TTYSession(session_id, self.config)
        return self.sessions[session_id]

    def reset_session(self, session_id: int = 0, reason: str = "") -> str:
        """Reset a specific session.

        Args:
            session_id: The session number (default: 0).
            reason: Optional reason for the reset.

        Returns:
            Confirmation message.
        """
        if session_id in self.sessions:
            return self.sessions[session_id].reset(reason)
        else:
            self.sessions[session_id] = TTYSession(session_id, self.config)
            return (
                f"Session {session_id} created."
                f" Reason: {reason or 'new session'}"
            )

    def cleanup_all(self) -> None:
        """Clean up all sessions."""
        for session in self.sessions.values():
            session._cleanup()
        self.sessions.clear()
