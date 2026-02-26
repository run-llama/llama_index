"""Helpers for extracting ``rubric_fn`` from model-generated code blocks."""

from __future__ import annotations

import re
import textwrap
from typing import Optional, Tuple


_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```", re.DOTALL
)

_DEF_RE = re.compile(
    r"(def\s+rubric_fn\s*\(.*?\).*?)(?=\ndef\s|\Z)", re.DOTALL
)

_EXPECTED_SIGNATURE = re.compile(
    r"def\s+rubric_fn\s*\(\s*input_text\s*(?::\s*str)?\s*,\s*response\s*(?::\s*str)?\s*\)"
)


def extract_code_blocks(text: str) -> list[str]:
    """Return all fenced code blocks from *text*."""
    return _FENCE_RE.findall(text)


def extract_rubric_fn_source(text: str) -> Optional[str]:
    """Extract the ``rubric_fn`` definition from model output.

    Searches fenced code blocks first, then falls back to the raw text.
    Returns the source code of the function, or ``None`` if not found.
    """
    # Try fenced code blocks first
    for block in extract_code_blocks(text):
        source = _find_rubric_fn_in_source(block)
        if source is not None:
            return source

    # Fallback: search raw text
    return _find_rubric_fn_in_source(text)


def _find_rubric_fn_in_source(source: str) -> Optional[str]:
    """Find a ``rubric_fn`` definition within a source string."""
    match = _DEF_RE.search(source)
    if match is None:
        return None

    fn_source = match.group(1).rstrip()

    # If the function is preceded by helper imports/code, include everything
    # from the start of the block up to the end of rubric_fn.
    preamble_end = match.start()
    preamble = source[:preamble_end].rstrip()
    if preamble:
        return preamble + "\n\n" + fn_source

    return fn_source


def validate_signature(source: str) -> Tuple[bool, Optional[str]]:
    """Check that *source* contains a ``rubric_fn(input_text, response)`` signature.

    Returns:
        (is_valid, error_message) — error_message is ``None`` when valid.
    """
    if "def rubric_fn" not in source:
        return False, "No `rubric_fn` definition found."

    if not _EXPECTED_SIGNATURE.search(source):
        return False, (
            "Signature mismatch: expected "
            "`def rubric_fn(input_text: str, response: str) -> float`."
        )

    return True, None


def compile_rubric_fn(source: str) -> Tuple[Optional[object], Optional[str]]:
    """Compile *source* and return the ``rubric_fn`` callable.

    Returns:
        (rubric_fn, error_message) — rubric_fn is ``None`` on failure.
    """
    is_valid, err = validate_signature(source)
    if not is_valid:
        return None, err

    namespace: dict = {}
    try:
        exec(textwrap.dedent(source), namespace)  # noqa: S102
    except Exception as exc:
        return None, f"Compilation error: {exc}"

    fn = namespace.get("rubric_fn")
    if fn is None:
        return None, "`rubric_fn` not found in compiled namespace."

    if not callable(fn):
        return None, "`rubric_fn` is not callable."

    return fn, None


def probe_callability(
    source: str,
    test_input: str = "hello",
    test_response: str = "world",
) -> Tuple[bool, Optional[str]]:
    """Compile and invoke ``rubric_fn`` with dummy inputs.

    Returns:
        (is_callable, error_message)
    """
    fn, err = compile_rubric_fn(source)
    if fn is None:
        return False, err

    try:
        result = fn(test_input, test_response)  # type: ignore[operator]
    except Exception as exc:
        return False, f"Runtime error: {exc}"

    if not isinstance(result, (int, float)):
        return False, f"Expected float return, got {type(result).__name__}."

    return True, None
