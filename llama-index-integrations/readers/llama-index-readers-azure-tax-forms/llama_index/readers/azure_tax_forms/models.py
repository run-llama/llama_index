"""
Data models for extracted tax form content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional


class TaxFormType(str, Enum):
    """IRS form types supported by the extraction pipeline."""

    FORM_1040 = "1040"
    W2 = "W-2"
    SCHEDULE_C = "Schedule C"
    SCHEDULE_E = "Schedule E"
    SCHEDULE_K1 = "Schedule K-1"
    FORM_1065 = "1065"
    FORM_1120 = "1120"
    FORM_1120S = "1120-S"
    UNKNOWN = "unknown"


@dataclass
class KvEntry:
    """
    A single key-value pair extracted from a tax form.

    Mirrors the Java KvEntry DTO.  The ``value`` is always a raw string
    as returned by Azure DI; callers that need a numeric value should use
    :meth:`value_as_decimal`.
    """

    key: str
    value: Optional[str]
    confidence: Optional[float]

    def value_as_decimal(self) -> Optional[Decimal]:
        """
        Parses ``value`` as a Decimal, stripping common formatting
        characters ($, commas, surrounding quotes, parentheses for
        negative values).

        Returns ``None`` when the value is absent or unparseable.
        """
        if not self.value:
            return None
        cleaned = (
            self.value.strip()
            .strip('"')
            .replace("$", "")
            .replace(",", "")
        )
        # Parentheses indicate a negative number: (4,200) → -4200
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    def __repr__(self) -> str:
        return f"KvEntry(key={self.key!r}, value={self.value!r}, confidence={self.confidence})"


@dataclass
class ExtractionResult:
    """
    The full output from a single document extraction.

    Attributes:
        document_id:   Caller-supplied identifier (file path, S3 key, UUID, etc.)
        form_type:     Detected or inferred form type.
        entries:       Ordered list of KV pairs extracted from the document.
        stage:         Which recovery stage produced the result
                       (``"STAGE-0"`` through ``"STAGE-2/3"``).
        di_calls:      Number of Azure Document Intelligence calls made.
        az_di_ms:      Total wall time spent inside Azure DI calls (ms).
        total_ms:      Total wall time for the full extraction (ms).
        error:         Set when extraction failed and ``entries`` is empty.
    """

    document_id: str
    form_type: TaxFormType = TaxFormType.UNKNOWN
    entries: list[KvEntry] = field(default_factory=list)
    stage: str = "STAGE-0"
    di_calls: int = 0
    az_di_ms: int = 0
    total_ms: int = 0
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def as_dict(self) -> dict:
        """Serialisable representation for LlamaIndex Document metadata."""
        return {
            "document_id": self.document_id,
            "form_type": self.form_type.value,
            "kv_count": len(self.entries),
            "stage": self.stage,
            "di_calls": self.di_calls,
            "az_di_ms": self.az_di_ms,
            "total_ms": self.total_ms,
            "error": self.error,
        }
