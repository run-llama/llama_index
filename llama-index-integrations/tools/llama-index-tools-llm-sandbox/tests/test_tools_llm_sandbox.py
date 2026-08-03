from unittest.mock import MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.llm_sandbox import DEFAULT_RUNTIME_CONFIGS, LLMSandboxToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in LLMSandboxToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert LLMSandboxToolSpec.spec_functions == ["execute_code"]


def test_default_runtime_configs_are_hardened():
    # Regression guard: these defaults are the reason this tool is safer than
    # executing on the host, so silently losing one should fail the build.
    assert DEFAULT_RUNTIME_CONFIGS["network_mode"] == "none"
    assert DEFAULT_RUNTIME_CONFIGS["cap_drop"] == ["ALL"]
    assert DEFAULT_RUNTIME_CONFIGS["security_opt"] == ["no-new-privileges:true"]
    assert "mem_limit" in DEFAULT_RUNTIME_CONFIGS
    assert "pids_limit" in DEFAULT_RUNTIME_CONFIGS


def test_default_runtime_configs_omit_unsupported_flags():
    # read_only makes Docker reject the code copy that SandboxSession performs
    # ("container rootfs is marked read-only"), so it must not be a default.
    assert "read_only" not in DEFAULT_RUNTIME_CONFIGS
    # cap_drop=ALL alone leaves the copied file unreadable; DAC_OVERRIDE is
    # added back deliberately.
    assert DEFAULT_RUNTIME_CONFIGS["cap_add"] == ["DAC_OVERRIDE"]


def test_to_tool_list_exposes_execute_code():
    tools = LLMSandboxToolSpec().to_tool_list()
    assert [t.metadata.name for t in tools] == ["execute_code"]


def _mock_session(exit_code=0, stdout="", stderr=""):
    result = MagicMock(exit_code=exit_code, stdout=stdout, stderr=stderr)
    session = MagicMock()
    session.run.return_value = result
    ctx = MagicMock()
    ctx.__enter__.return_value = session
    ctx.__exit__.return_value = False
    return ctx, session


@patch("llm_sandbox.SandboxSession")
def test_execute_code_returns_stdout(mock_cls):
    ctx, _ = _mock_session(stdout="42\n")
    mock_cls.return_value = ctx
    assert LLMSandboxToolSpec().execute_code("print(6 * 7)") == "42"


@patch("llm_sandbox.SandboxSession")
def test_execute_code_reports_failure(mock_cls):
    ctx, _ = _mock_session(exit_code=1, stderr="Traceback: boom")
    mock_cls.return_value = ctx
    out = LLMSandboxToolSpec().execute_code("raise ValueError")
    assert out.startswith("exit 1")
    assert "boom" in out


@patch("llm_sandbox.SandboxSession")
def test_execute_code_handles_empty_output(mock_cls):
    ctx, _ = _mock_session(stdout="   ")
    mock_cls.return_value = ctx
    assert LLMSandboxToolSpec().execute_code("x = 1") == "(no output)"


@patch("llm_sandbox.SandboxSession")
def test_hardening_is_passed_to_the_session(mock_cls):
    ctx, _ = _mock_session(stdout="ok")
    mock_cls.return_value = ctx
    LLMSandboxToolSpec().execute_code("print('ok')")
    assert mock_cls.call_args.kwargs["runtime_configs"] == DEFAULT_RUNTIME_CONFIGS
    # Without this the image is deleted on close and re-pulled every call.
    assert mock_cls.call_args.kwargs["keep_template"] is True


@patch("llm_sandbox.SandboxSession")
def test_runtime_configs_can_be_overridden(mock_cls):
    ctx, _ = _mock_session(stdout="ok")
    mock_cls.return_value = ctx
    LLMSandboxToolSpec(runtime_configs={"network_mode": "bridge"}).execute_code("print(1)")
    assert mock_cls.call_args.kwargs["runtime_configs"] == {"network_mode": "bridge"}


def test_tool_exposes_only_code():
    # A  parameter here would let the model pick arbitrary PyPI
    # packages, which executes setup.py at install time. Keep the surface to one
    # argument.
    tool = LLMSandboxToolSpec().to_tool_list()[0]
    assert set(tool.metadata.get_parameters_dict()["properties"]) == {"code"}


@patch("llm_sandbox.SandboxSession")
def test_lang_and_timeout_are_forwarded(mock_cls):
    ctx, session = _mock_session(stdout="ok")
    mock_cls.return_value = ctx
    LLMSandboxToolSpec(lang="ruby", timeout=5.0).execute_code("puts 1")
    assert mock_cls.call_args.kwargs["lang"] == "ruby"
    assert session.run.call_args.kwargs["timeout"] == 5.0
