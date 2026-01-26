"""Tests for CausalReader."""

from unittest.mock import MagicMock, patch

import pytest


def test_causal_reader_import():
    """Test that CausalReader can be imported."""
    from llama_index.readers.causal import CausalReader

    assert CausalReader is not None


def test_causal_reader_class_name():
    """Test class name."""
    with patch("llama_index.readers.causal.base.dotcausal"):
        from llama_index.readers.causal import CausalReader

        assert CausalReader.class_name() == "CausalReader"


def test_causal_reader_load_data():
    """Test loading data from .causal file."""
    with patch("llama_index.readers.causal.base.dotcausal"):
        with patch(
            "llama_index.readers.causal.base.DotCausalReader"
        ) as mock_reader_class:
            with patch("pathlib.Path.exists", return_value=True):
                from llama_index.readers.causal import CausalReader

                # Setup mock
                mock_reader = MagicMock()
                mock_reader.search.return_value = [
                    {
                        "trigger": "COVID",
                        "mechanism": "causes",
                        "outcome": "fatigue",
                        "confidence": 0.9,
                        "is_inferred": False,
                        "source": "paper1.pdf",
                    },
                    {
                        "trigger": "SARS-CoV-2",
                        "mechanism": "damages",
                        "outcome": "mitochondria",
                        "confidence": 0.85,
                        "is_inferred": True,
                        "provenance": ["t1", "t2"],
                    },
                ]
                mock_reader_class.return_value = mock_reader

                # Test
                reader = CausalReader()
                docs = reader.load_data("test.causal", query="COVID")

                assert len(docs) == 2
                assert "[EXPLICIT]" in docs[0].text
                assert "[INFERRED]" in docs[1].text
                assert docs[0].metadata["trigger"] == "COVID"
                assert docs[1].metadata["is_inferred"] is True


def test_causal_reader_confidence_filter():
    """Test confidence filtering."""
    with patch("llama_index.readers.causal.base.dotcausal"):
        with patch(
            "llama_index.readers.causal.base.DotCausalReader"
        ) as mock_reader_class:
            with patch("pathlib.Path.exists", return_value=True):
                from llama_index.readers.causal import CausalReader

                mock_reader = MagicMock()
                mock_reader.search.return_value = [
                    {
                        "trigger": "A",
                        "mechanism": "causes",
                        "outcome": "B",
                        "confidence": 0.9,
                        "is_inferred": False,
                    },
                    {
                        "trigger": "C",
                        "mechanism": "causes",
                        "outcome": "D",
                        "confidence": 0.3,  # Below threshold
                        "is_inferred": False,
                    },
                ]
                mock_reader_class.return_value = mock_reader

                reader = CausalReader(min_confidence=0.5)
                docs = reader.load_data("test.causal", query="test")

                assert len(docs) == 1
                assert docs[0].metadata["confidence"] == 0.9


def test_causal_reader_exclude_inferred():
    """Test excluding inferred triplets."""
    with patch("llama_index.readers.causal.base.dotcausal"):
        with patch(
            "llama_index.readers.causal.base.DotCausalReader"
        ) as mock_reader_class:
            with patch("pathlib.Path.exists", return_value=True):
                from llama_index.readers.causal import CausalReader

                mock_reader = MagicMock()
                mock_reader.search.return_value = [
                    {
                        "trigger": "A",
                        "mechanism": "causes",
                        "outcome": "B",
                        "confidence": 0.9,
                        "is_inferred": False,
                    },
                    {
                        "trigger": "C",
                        "mechanism": "causes",
                        "outcome": "D",
                        "confidence": 0.85,
                        "is_inferred": True,
                    },
                ]
                mock_reader_class.return_value = mock_reader

                reader = CausalReader(include_inferred=False)
                docs = reader.load_data("test.causal", query="test")

                assert len(docs) == 1
                assert docs[0].metadata["is_inferred"] is False
