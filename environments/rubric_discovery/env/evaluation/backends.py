"""Execution backends for running candidate rubric_fn code in isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from abc import ABC, abstractmethod
from typing import List, Optional

from environments.rubric_discovery.env.types import (
    EvaluationResult,
    ExecutionBackend,
    LabeledExample,
)


class BaseBackend(ABC):
    """Abstract base for rubric_fn execution backends."""

    @abstractmethod
    def execute(
        self,
        source: str,
        examples: List[LabeledExample],
        timeout_s: int = 10,
    ) -> EvaluationResult:
        """Execute *source* against *examples* and return predictions."""


class SubprocessBackend(BaseBackend):
    """Execute candidate rubric_fn in a local subprocess."""

    def execute(
        self,
        source: str,
        examples: List[LabeledExample],
        timeout_s: int = 10,
    ) -> EvaluationResult:
        """Run rubric_fn in an isolated subprocess."""
        harness = _build_harness(source, examples)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(harness)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return EvaluationResult(
                labels=[e.score for e in examples],
                valid=False,
                error=f"Execution timed out after {timeout_s}s",
            )
        except Exception as exc:
            return EvaluationResult(
                labels=[e.score for e in examples],
                valid=False,
                error=f"Subprocess error: {exc}",
            )
        finally:
            os.unlink(script_path)

        if result.returncode != 0:
            return EvaluationResult(
                labels=[e.score for e in examples],
                valid=False,
                error=f"Process exited with code {result.returncode}: {result.stderr[:500]}",
            )

        try:
            predictions = json.loads(result.stdout.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            return EvaluationResult(
                labels=[e.score for e in examples],
                valid=False,
                error=f"Output parsing error: {exc}. stdout={result.stdout[:200]}",
            )

        return EvaluationResult(
            predictions=predictions,
            labels=[e.score for e in examples],
            valid=True,
        )


class SandboxBackend(BaseBackend):
    """Execute candidate rubric_fn in a Prime Sandbox.

    Requires ``PRIME_API_KEY`` to be set. Falls back to subprocess if
    the sandbox client is unavailable.
    """

    def execute(
        self,
        source: str,
        examples: List[LabeledExample],
        timeout_s: int = 10,
    ) -> EvaluationResult:
        """Run rubric_fn in a Prime Sandbox."""
        try:
            from prime_sandbox import SandboxClient  # type: ignore[import-untyped]
        except ImportError:
            return SubprocessBackend().execute(source, examples, timeout_s)

        api_key = os.environ.get("PRIME_API_KEY")
        if not api_key:
            return SubprocessBackend().execute(source, examples, timeout_s)

        harness = _build_harness(source, examples)

        try:
            client = SandboxClient(api_key=api_key)
            result = client.run_code(harness, timeout=timeout_s)
        except Exception as exc:
            return EvaluationResult(
                labels=[e.score for e in examples],
                valid=False,
                error=f"Sandbox error: {exc}",
            )

        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        exit_code = getattr(result, "exit_code", -1)

        if exit_code != 0:
            return EvaluationResult(
                labels=[e.score for e in examples],
                valid=False,
                error=f"Sandbox exited {exit_code}: {stderr[:500]}",
            )

        try:
            predictions = json.loads(stdout.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            return EvaluationResult(
                labels=[e.score for e in examples],
                valid=False,
                error=f"Output parsing error: {exc}",
            )

        return EvaluationResult(
            predictions=predictions,
            labels=[e.score for e in examples],
            valid=True,
        )


def resolve_backend(backend: ExecutionBackend) -> BaseBackend:
    """Return the appropriate backend instance based on config."""
    if backend == ExecutionBackend.SANDBOX:
        return SandboxBackend()
    if backend == ExecutionBackend.SUBPROCESS:
        return SubprocessBackend()

    # AUTO: use sandbox if PRIME_API_KEY is set
    if os.environ.get("PRIME_API_KEY"):
        return SandboxBackend()
    return SubprocessBackend()


def _build_harness(source: str, examples: List[LabeledExample]) -> str:
    """Build a self-contained Python script that runs rubric_fn on examples."""
    examples_json = json.dumps(
        [{"input_text": e.input_text, "response": e.response} for e in examples]
    )

    return textwrap.dedent(f"""\
        import json
        import sys

        # --- Candidate rubric_fn ---
        {textwrap.indent(source, "        ").strip()}

        # --- Evaluation harness ---
        _examples = json.loads('''{examples_json}''')
        _predictions = []
        for _ex in _examples:
            try:
                _score = rubric_fn(_ex["input_text"], _ex["response"])
                _score = float(_score)
                _score = max(0.0, min(1.0, _score))
            except Exception:
                _score = 0.0
            _predictions.append(_score)
        print(json.dumps(_predictions))
    """)
