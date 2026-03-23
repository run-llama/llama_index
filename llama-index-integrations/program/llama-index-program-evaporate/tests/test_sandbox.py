"""Tests for the sandboxed execution environment in EvaporateExtractor."""

import pytest

from llama_index.program.evaporate.extractor import (
    _build_sandbox,
    _validate_generated_code,
)


# ---------------------------------------------------------------------------
# _validate_generated_code
# ---------------------------------------------------------------------------


def test_validate_allows_safe_code():
    """Normal extraction function should pass validation."""
    code = (
        "def get_name_field(text: str):\n"
        '    match = re.search(r"Name: (.+)", text)\n'
        "    if match:\n"
        "        return match.group(1)\n"
        '    return ""\n'
    )
    # Should not raise
    _validate_generated_code(code)


def test_validate_allows_re_import():
    """Import re is used in the prompt template and should be allowed."""
    code = "import re\nx = re.search(r'a', 'a')\n"
    _validate_generated_code(code)


def test_validate_rejects_os_import():
    code = "import os\nos.system('echo pwned')\n"
    with pytest.raises(RuntimeError, match="imports 'os'"):
        _validate_generated_code(code)


def test_validate_rejects_subprocess_import():
    code = "import subprocess\nsubprocess.run(['ls'])\n"
    with pytest.raises(RuntimeError, match="imports 'subprocess'"):
        _validate_generated_code(code)


def test_validate_rejects_from_import():
    code = "from os.path import join\n"
    with pytest.raises(RuntimeError, match="imports from 'os.path'"):
        _validate_generated_code(code)


def test_validate_rejects_dunder_name():
    code = "x = __import__('os')\n"
    with pytest.raises(RuntimeError, match="dunder name '__import__'"):
        _validate_generated_code(code)


def test_validate_rejects_dunder_attribute():
    code = "x = ''.__class__.__bases__\n"
    with pytest.raises(RuntimeError, match="dunder attribute"):
        _validate_generated_code(code)


# ---------------------------------------------------------------------------
# _build_sandbox
# ---------------------------------------------------------------------------


def test_sandbox_has_re_module():
    sandbox = _build_sandbox("hello world")
    assert sandbox["re"] is not None
    assert sandbox["re"].search(r"hello", "hello world") is not None


def test_sandbox_has_node_text():
    sandbox = _build_sandbox("test text")
    assert sandbox["node_text"] == "test text"


def test_sandbox_blocks_open():
    sandbox = _build_sandbox("")
    with pytest.raises(NameError):
        exec("f = open('/etc/passwd')", sandbox)


def test_sandbox_blocks_eval():
    sandbox = _build_sandbox("")
    with pytest.raises(NameError):
        exec("eval('1+1')", sandbox)


def test_sandbox_blocks_exec_builtin():
    sandbox = _build_sandbox("")
    with pytest.raises(NameError):
        exec("exec('x=1')", sandbox)


def test_sandbox_blocks_unrestricted_import():
    sandbox = _build_sandbox("")
    with pytest.raises(ImportError, match="not allowed in the sandbox"):
        exec("import os", sandbox)


def test_sandbox_allows_re_import_at_runtime():
    sandbox = _build_sandbox("")
    exec("import re\nx = re.search(r'a', 'abc')", sandbox)
    assert sandbox["x"] is not None


def test_sandbox_allows_stdlib_imports_at_runtime():
    """Stdlib modules in the allowlist should be importable at runtime."""
    sandbox = _build_sandbox("")
    exec("import datetime\nx = datetime.date(2026, 1, 1).isoformat()", sandbox)
    assert sandbox["x"] == "2026-01-01"

    sandbox = _build_sandbox("")
    exec("import collections\nx = collections.Counter('aab')", sandbox)
    assert sandbox["x"]["a"] == 2


def test_sandbox_exec_extraction_function():
    """End-to-end: define and call a function inside the sandbox."""
    fn_str = (
        "def get_name_field(text):\n"
        '    match = re.search(r"Name: (.+)", text)\n'
        "    if match:\n"
        "        return match.group(1).strip()\n"
        '    return ""\n'
    )
    sandbox = _build_sandbox("Name: Alice Johnson")
    exec(fn_str, sandbox)
    exec("__result__ = get_name_field(node_text)", sandbox)
    assert sandbox["__result__"] == "Alice Johnson"


def test_sandbox_basic_builtins_available():
    """Generated functions should be able to use common builtins."""
    sandbox = _build_sandbox("")
    exec("x = len([1, 2, 3])", sandbox)
    assert sandbox["x"] == 3

    exec("y = int('42')", sandbox)
    assert sandbox["y"] == 42

    exec("z = sorted([3, 1, 2])", sandbox)
    assert sandbox["z"] == [1, 2, 3]
