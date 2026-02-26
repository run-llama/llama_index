"""Tests for env/candidate/extractor.py: rubric_fn extraction helpers."""

import pytest

from environments.rubric_discovery.env.candidate.extractor import (
    compile_rubric_fn,
    extract_code_blocks,
    extract_rubric_fn_source,
    probe_callability,
    validate_signature,
)


class TestExtractCodeBlocks:
    def test_single_block(self) -> None:
        text = "```python\nprint('hello')\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "print('hello')" in blocks[0]

    def test_multiple_blocks(self) -> None:
        text = "```python\nblock1\n```\ntext\n```\nblock2\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2

    def test_no_blocks(self) -> None:
        text = "no code here"
        assert extract_code_blocks(text) == []


class TestExtractRubricFnSource:
    def test_from_code_block(self) -> None:
        text = '''Here is my rubric:
```python
def rubric_fn(input_text, response):
    return 0.5
```'''
        source = extract_rubric_fn_source(text)
        assert source is not None
        assert "def rubric_fn" in source

    def test_from_raw_text(self) -> None:
        text = """
def rubric_fn(input_text, response):
    if "correct" in response:
        return 1.0
    return 0.0
"""
        source = extract_rubric_fn_source(text)
        assert source is not None
        assert "correct" in source

    def test_with_preamble(self) -> None:
        text = '''```python
import re

def rubric_fn(input_text, response):
    return len(response) / 100
```'''
        source = extract_rubric_fn_source(text)
        assert source is not None
        assert "import re" in source

    def test_no_rubric_fn(self) -> None:
        text = "def other_function(x): return x"
        assert extract_rubric_fn_source(text) is None


class TestValidateSignature:
    def test_valid_untyped(self) -> None:
        source = "def rubric_fn(input_text, response): return 0.5"
        ok, err = validate_signature(source)
        assert ok
        assert err is None

    def test_valid_typed(self) -> None:
        source = "def rubric_fn(input_text: str, response: str) -> float:\n    return 0.5"
        ok, err = validate_signature(source)
        assert ok

    def test_missing_function(self) -> None:
        source = "def other(x): return x"
        ok, err = validate_signature(source)
        assert not ok
        assert "No `rubric_fn` definition" in err

    def test_wrong_params(self) -> None:
        source = "def rubric_fn(text): return 0.5"
        ok, err = validate_signature(source)
        assert not ok
        assert "Signature mismatch" in err


class TestCompileRubricFn:
    def test_simple(self) -> None:
        source = "def rubric_fn(input_text, response): return 0.5"
        fn, err = compile_rubric_fn(source)
        assert fn is not None
        assert err is None
        assert fn("a", "b") == 0.5

    def test_syntax_error(self) -> None:
        source = "def rubric_fn(input_text, response):\n    return ++++"
        fn, err = compile_rubric_fn(source)
        assert fn is None
        assert "Compilation error" in err

    def test_not_callable(self) -> None:
        source = "def rubric_fn(input_text, response): return 0.5\nrubric_fn = 42"
        fn, err = compile_rubric_fn(source)
        assert fn is None
        assert "not callable" in err


class TestProbeCallability:
    def test_callable(self) -> None:
        source = "def rubric_fn(input_text, response): return 0.5"
        ok, err = probe_callability(source)
        assert ok
        assert err is None

    def test_runtime_error(self) -> None:
        source = "def rubric_fn(input_text, response): return 1/0"
        ok, err = probe_callability(source)
        assert not ok
        assert "Runtime error" in err

    def test_wrong_return_type(self) -> None:
        source = 'def rubric_fn(input_text, response): return "not a float"'
        ok, err = probe_callability(source)
        assert not ok
        assert "Expected float" in err
