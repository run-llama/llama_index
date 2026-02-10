"""String manipulation utilities for code execution output processing."""

import re


# Pre-compiled regex patterns
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_PROMPT_PATTERNS = [
    re.compile(r"[\$#%>]\s*$"),  # Common shell prompts
    re.compile(r"\w+@[\w.-]+[:\w~\/]*[\$#]\s*$"),  # user@host:path$
    re.compile(r"\(.*\)\s*[\$#%>]\s*$"),  # (venv) $
    re.compile(r">>>\s*$"),  # Python REPL
    re.compile(r"\.\.\.\s*$"),  # Python continuation
    re.compile(r"In\s*\[\d+\]:\s*$"),  # IPython prompt
]

_DIALOG_PATTERNS = [
    re.compile(r"\[Y/n\]", re.IGNORECASE),
    re.compile(r"\[y/N\]", re.IGNORECASE),
    re.compile(r"\(yes/no\)", re.IGNORECASE),
    re.compile(r"\(y/n\)", re.IGNORECASE),
    re.compile(r"Press \w+ to continue", re.IGNORECASE),
    re.compile(r"Do you want to continue", re.IGNORECASE),
    re.compile(r"Are you sure", re.IGNORECASE),
    re.compile(r"Enter password", re.IGNORECASE),
    re.compile(r"Password:\s*$", re.IGNORECASE),
    re.compile(r"passphrase", re.IGNORECASE),
]


def clean_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


def detect_prompt(text: str) -> bool:
    """Detect if the text ends with a shell prompt."""
    cleaned = clean_ansi(text.rstrip())
    if not cleaned:
        return False
    # Check last line only
    last_line = cleaned.split("\n")[-1]
    for pattern in _PROMPT_PATTERNS:
        if pattern.search(last_line):
            return True
    return False


def detect_ipython_prompt(text: str) -> bool:
    """Detect if text ends with an IPython In[N]: prompt."""
    cleaned = clean_ansi(text.rstrip())
    if not cleaned:
        return False
    return bool(re.search(r"In\s*\[\d+\]:\s*$", cleaned))


def detect_dialog(text: str) -> bool:
    """Detect interactive dialog prompts like [Y/n], (yes/no), etc."""
    cleaned = clean_ansi(text)
    for pattern in _DIALOG_PATTERNS:
        if pattern.search(cleaned):
            return True
    return False


def truncate_output(text: str, max_chars: int = 50000) -> str:
    """Truncate output if it exceeds max characters, keeping start and end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [truncated {len(text) - max_chars} characters] ...\n\n"
        + text[-half:]
    )
