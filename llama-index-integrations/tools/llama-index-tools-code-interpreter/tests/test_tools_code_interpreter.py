from unittest.mock import MagicMock, patch
import subprocess
import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.code_interpreter import (
    CodeInterpreterToolSpec,
    VettoCodeInterpreterToolSpec,
)
from llama_index.tools.code_interpreter.base import (
    _build_clean_sandboxed_env,
    BLOCKED_SECRET_ENV_PREFIXES,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in CodeInterpreterToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_vetto_subclass_inheritance():
    names_of_base_classes = [b.__name__ for b in VettoCodeInterpreterToolSpec.__mro__]
    assert CodeInterpreterToolSpec.__name__ in names_of_base_classes
    assert BaseToolSpec.__name__ in names_of_base_classes

    spec = VettoCodeInterpreterToolSpec(timeout=15, net_mode="off")
    assert spec.sandbox == "vetto"
    assert spec.timeout == 15
    assert spec.net_mode == "off"
    assert spec.fail_closed is True


def test_default_unconfined_execution():
    spec = CodeInterpreterToolSpec()
    assert spec.sandbox is None
    result = spec.code_interpreter("print('hello unconfined')")
    assert "StdOut:\nhello unconfined" in result
    assert "StdErr:\n" in result


def test_vetto_missing_binary_fail_closed():
    spec = CodeInterpreterToolSpec(sandbox="vetto", fail_closed=True)
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            spec.code_interpreter("print(1)")
        assert "Vetto sandbox execution requested" in str(exc_info.value)
        assert "not found in PATH" in str(exc_info.value)


def test_vetto_missing_binary_fallback_when_not_fail_closed():
    spec = CodeInterpreterToolSpec(sandbox="vetto", fail_closed=False)
    with patch("shutil.which", return_value=None):
        result = spec.code_interpreter("print('fallback run')")
        assert "StdOut:\nfallback run" in result


def test_vetto_command_construction():
    spec = CodeInterpreterToolSpec(
        sandbox="vetto",
        working_dir="/tmp/agent-work",
        timeout=45,
        net_mode="allowlist:api.openai.com",
    )
    with patch("shutil.which", return_value="/usr/local/bin/vetto"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=b"result ok\n", stderr=b"")
        res = spec.code_interpreter("print('secure')")

        assert mock_run.called
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "/usr/local/bin/vetto"
        assert cmd[1] == "run"
        assert "--workspace" in cmd
        assert "/tmp/agent-work" in cmd
        assert "--timeout=45s" in cmd
        assert "--net=allowlist:api.openai.com" in cmd
        assert "--" in cmd
        assert "print('secure')" in cmd[-1]
        assert kwargs.get("cwd") == "/tmp/agent-work"
        assert kwargs.get("timeout") == 45
        assert "StdOut:\nresult ok\n" in res


def test_vetto_clean_env_strips_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("DATABASE_PASSWORD", "db-pass")
    monkeypatch.setenv("SAFE_VAR", "visible-value")

    clean_env = _build_clean_sandboxed_env()
    assert "OPENAI_API_KEY" not in clean_env
    assert "ANTHROPIC_API_KEY" not in clean_env
    assert "AWS_SECRET_ACCESS_KEY" not in clean_env
    assert "DATABASE_PASSWORD" not in clean_env
    assert clean_env.get("SAFE_VAR") == "visible-value"


def test_vetto_timeout_handling():
    spec = CodeInterpreterToolSpec(sandbox="vetto", timeout=5)
    with patch("shutil.which", return_value="/usr/bin/vetto"), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["vetto"], timeout=5, output=b"partial", stderr=b"timeout err")):
        res = spec.code_interpreter("import time; time.sleep(10)")
        assert "partial" in res
        assert "Command timed out after 5 seconds." in res
