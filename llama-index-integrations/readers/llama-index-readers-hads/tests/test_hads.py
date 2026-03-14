"""Tests for HADSReader."""

import tempfile
from pathlib import Path

import pytest

from llama_index.readers.hads import HADSReader

SAMPLE_HADS = """\
# My Module

## AI READING INSTRUCTION
Load SPEC blocks for implementation details. Load NOTE for context.

## Core Logic

**[SPEC]**
This is a specification block with implementation details.

**[NOTE]**
This is a note with background context.

**[BUG memory-leak]**
Memory leak in the connection pool under load.

**[?]**
Should we use async here?
"""

SPEC_ONLY = """\
**[SPEC]**
Only block here.
"""

MULTI_SECTION = """\
# Top Level

## Section A

**[SPEC]**
Block in section A.

## Section B

**[SPEC]**
Block in section B.
"""


def _write(content: str, suffix: str = ".md") -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def test_default_loads_spec_only():
    reader = HADSReader()
    docs = reader.load_data(_write(SAMPLE_HADS))
    assert len(docs) == 1
    assert "[SPEC]" in docs[0].text
    assert "[NOTE]" not in docs[0].text


def test_spec_and_note():
    reader = HADSReader(block_types=["SPEC", "NOTE"])
    docs = reader.load_data(_write(SAMPLE_HADS))
    tags = [d.metadata["block_tag"] for d in docs]
    assert "SPEC" in tags
    assert "NOTE" in tags
    assert len([t for t in tags if t.startswith("BUG")]) == 0


def test_all_block_types():
    reader = HADSReader(block_types=["SPEC", "NOTE", "BUG", "?"])
    docs = reader.load_data(_write(SAMPLE_HADS))
    assert len(docs) == 4


def test_bug_prefix_matching():
    reader = HADSReader(block_types=["BUG"])
    docs = reader.load_data(_write(SAMPLE_HADS))
    assert len(docs) == 1
    assert "memory-leak" in docs[0].text


def test_manifest_extracted_to_metadata():
    reader = HADSReader()
    docs = reader.load_data(_write(SAMPLE_HADS))
    assert "manifest" in docs[0].metadata
    assert "SPEC" in docs[0].metadata["manifest"]


def test_metadata_fields():
    reader = HADSReader()
    docs = reader.load_data(_write(SPEC_ONLY))
    meta = docs[0].metadata
    assert meta["hads"] is True
    assert meta["block_types"] == ["SPEC"]
    assert "source" in meta
    assert meta["blocks_found"] == 1


def test_section_headers_in_text():
    reader = HADSReader(include_section_headers=True)
    docs = reader.load_data(_write(MULTI_SECTION))
    assert len(docs) == 2
    assert "Section A" in docs[0].text
    assert "Section B" in docs[1].text


def test_no_section_headers_when_disabled():
    reader = HADSReader(include_section_headers=False)
    docs = reader.load_data(_write(MULTI_SECTION))
    assert "Section A" not in docs[0].text


def test_missing_file_raises():
    reader = HADSReader()
    with pytest.raises(ValueError, match="not found"):
        reader.load_data(Path("/nonexistent/file.hads.md"))


def test_no_matching_blocks_returns_empty_doc():
    reader = HADSReader(block_types=["BUG"])
    docs = reader.load_data(_write(SPEC_ONLY))
    assert len(docs) == 1
    assert docs[0].text == ""
