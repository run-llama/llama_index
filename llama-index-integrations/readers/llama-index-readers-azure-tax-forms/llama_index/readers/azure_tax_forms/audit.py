"""
Extraction audit logger — file-bounded, no stdout.

Writes one row per document to ``logs/extraction-audit.log``.

Output format (pipe-delimited):
    TIMESTAMP           | DOCUMENT ID                | FORM TYPE    | STAGE        | KV_PAIRS | DI_CALLS | AZ_DI_MS | TOTAL_MS
    2026-06-06 10:15:23 | w2_john_doe_2023.pdf       | W-2          | STAGE-0      |       18 |        1 |   1823ms |   1902ms
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from llama_index.readers.azure_tax_forms.models import ExtractionResult

_AUDIT_LOGGER_NAME = "azure_tax_forms.extraction_audit"
_CONFIGURED = False


def configure_audit_logger(
    log_dir: str = "logs",
    max_days: int = 30,
    enabled: bool = True,
) -> None:
    """
    Configure the extraction audit logger.

    Call once at application startup.  Safe to call multiple times — only
    the first call has effect.

    Args:
        log_dir:  Directory for the rolling log file.
        max_days: Days of log retention.
        enabled:  Set ``False`` to disable audit logging entirely (e.g. in tests).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    audit_logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    audit_logger.propagate = False  # never write to stdout / root logger

    if not enabled:
        audit_logger.addHandler(logging.NullHandler())
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, "extraction-audit.log")

    handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        backupCount=max_days,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)


_HEADER_WRITTEN = False


def write_header() -> None:
    """Write the column header row.  Call once per batch/session."""
    global _HEADER_WRITTEN
    if _HEADER_WRITTEN:
        return
    _HEADER_WRITTEN = True
    row = _fmt_row(
        "TIMESTAMP", "DOCUMENT ID", "FORM TYPE", "STAGE", "KV_PAIRS",
        "DI_CALLS", "AZ_DI_MS", "TOTAL_MS",
    )
    sep = "-" * len(row)
    _logger().info(row)
    _logger().info(sep)


def record(result: ExtractionResult) -> None:
    """Write one audit row for a completed extraction."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = _fmt_row(
        ts,
        result.document_id,
        result.form_type.value,
        result.stage,
        str(result.entries.__len__()),
        str(result.di_calls) if result.di_calls else "-",
        f"{result.az_di_ms}ms" if result.az_di_ms else "-",
        f"{result.total_ms}ms",
    )
    _logger().info(row)


# column widths
_W = {"ts": 19, "doc": 40, "form": 14, "stage": 14, "kv": 8, "calls": 8, "di": 10, "total": 10}


def _fmt_row(*cols: str) -> str:
    widths = list(_W.values())
    parts = []
    for i, col in enumerate(cols):
        w = widths[i] if i < len(widths) else 10
        parts.append(f"{col:<{w}}" if i < 3 else f"{col:>{w}}")
    return " | ".join(parts)


def _logger() -> logging.Logger:
    return logging.getLogger(_AUDIT_LOGGER_NAME)
