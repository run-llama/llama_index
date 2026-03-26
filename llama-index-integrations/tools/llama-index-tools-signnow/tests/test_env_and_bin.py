import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Mapping, cast
from llama_index.tools.signnow.base import SignNowMCPToolSpec, EXPECTED_SIGNNOW_KEYS


def test_from_env_requires_auth() -> None:
    with pytest.raises(ValueError):
        SignNowMCPToolSpec.from_env(env_overrides={}, require_in_path=False)


@patch("shutil.which", return_value=None)
def test_bin_not_found(mock_which: MagicMock) -> None:
    with pytest.raises(FileNotFoundError):
        SignNowMCPToolSpec.from_env(
            env_overrides={"SIGNNOW_TOKEN": "tok"}, require_in_path=True
        )


@patch("shutil.which", return_value="/usr/local/bin/sn-mcp")
@patch("llama_index.tools.signnow.base.BasicMCPClient")
def test_auth_token_and_env_filtering(
    mock_client: MagicMock, mock_which: MagicMock
) -> None:
    spec = SignNowMCPToolSpec.from_env(
        env_overrides=cast(
            Mapping[str, str],
            {
                "SIGNNOW_TOKEN": "dummy",
                "IRRELEVANT": "skip",
                "SIGNNOW_API_BASE": None,  # should be filtered out
            },
        ),
        require_in_path=True,
    )
    # Ensure client constructed with only expected env keys
    called_env = mock_client.call_args.kwargs["env"]
    assert set(called_env.keys()).issubset(EXPECTED_SIGNNOW_KEYS)
    assert "SIGNNOW_TOKEN" in called_env and "IRRELEVANT" not in called_env
    # Spec should be created successfully
    assert isinstance(spec, SignNowMCPToolSpec)


@patch("shutil.which", return_value="/usr/local/bin/sn-mcp")
@patch("llama_index.tools.signnow.base.BasicMCPClient")
def test_auth_basic_credentials(mock_client: MagicMock, mock_which: MagicMock) -> None:
    SignNowMCPToolSpec.from_env(
        env_overrides={
            "SIGNNOW_USER_EMAIL": "u@example.com",
            "SIGNNOW_PASSWORD": "pass",
            "SIGNNOW_API_BASIC_TOKEN": "basic",
        },
        require_in_path=True,
    )
    called_env = mock_client.call_args.kwargs["env"]
    assert called_env.get("SIGNNOW_USER_EMAIL") == "u@example.com"
    assert called_env.get("SIGNNOW_PASSWORD") == "pass"
    assert called_env.get("SIGNNOW_API_BASIC_TOKEN") == "basic"


@patch("shutil.which", return_value="/usr/local/bin/sn-mcp")
@patch("llama_index.tools.signnow.base.BasicMCPClient")
def test_auth_both_token_and_basic(
    mock_client: MagicMock, mock_which: MagicMock
) -> None:
    SignNowMCPToolSpec.from_env(
        env_overrides={
            "SIGNNOW_TOKEN": "tok",
            "SIGNNOW_USER_EMAIL": "u@example.com",
            "SIGNNOW_PASSWORD": "pass",
            "SIGNNOW_API_BASIC_TOKEN": "basic",
        },
        require_in_path=True,
    )
    called_env = mock_client.call_args.kwargs["env"]
    # Both sets are allowed; presence is sufficient
    assert called_env.get("SIGNNOW_TOKEN") == "tok"
    assert called_env.get("SIGNNOW_USER_EMAIL") == "u@example.com"


def test_bin_resolution_with_env_var() -> None:
    with patch.dict(os.environ, {"SIGNNOW_MCP_BIN": "custom-bin"}, clear=False):
        with patch("shutil.which", return_value="/opt/custom-bin") as mock_which:
            with patch("llama_index.tools.signnow.base.BasicMCPClient") as mock_client:
                SignNowMCPToolSpec.from_env(
                    env_overrides={"SIGNNOW_TOKEN": "tok"}, require_in_path=True
                )
                # Ensure resolver tried env var and used absolute path
                mock_which.assert_called_with("custom-bin")
                assert mock_client.call_args.args[0] == "/opt/custom-bin"


@patch("llama_index.tools.signnow.base.BasicMCPClient")
def test_bin_resolution_with_explicit_param(mock_client: MagicMock) -> None:
    with patch("shutil.which", return_value="/bin/local-sn") as mock_which:
        SignNowMCPToolSpec.from_env(
            env_overrides={"SIGNNOW_TOKEN": "tok"}, bin="local-sn", require_in_path=True
        )
        mock_which.assert_called_with("local-sn")
        assert mock_client.call_args.args[0] == "/bin/local-sn"


@patch("llama_index.tools.signnow.base.BasicMCPClient")
def test_bin_candidate_when_not_required(mock_client: MagicMock) -> None:
    # PATH resolution fails; should fall back to candidate without raising
    with patch("shutil.which", return_value=None):
        SignNowMCPToolSpec.from_env(
            env_overrides={"SIGNNOW_TOKEN": "tok"}, require_in_path=False
        )
        # Default candidate is "sn-mcp"
        assert mock_client.call_args.args[0] == "sn-mcp"


@patch("shutil.which", return_value="/usr/local/bin/sn-mcp")
@patch("llama_index.tools.signnow.base.BasicMCPClient")
def test_cmd_and_args_passed_to_client(
    mock_client: MagicMock, mock_which: MagicMock
) -> None:
    SignNowMCPToolSpec.from_env(
        env_overrides={"SIGNNOW_TOKEN": "tok"},
        cmd="serve",
        args=["--flag", "v"],
        require_in_path=True,
    )
    assert mock_client.call_args.kwargs["args"] == ["serve", "--flag", "v"]


@patch("shutil.which", return_value="/usr/local/bin/sn-mcp")
@patch("llama_index.tools.signnow.base.McpToolSpec")
@patch("llama_index.tools.signnow.base.BasicMCPClient")
def test_allowed_tools_and_include_resources_propagated(
    mock_client: MagicMock, mock_mcp_spec: MagicMock, mock_which: MagicMock
) -> None:
    SignNowMCPToolSpec.from_env(
        env_overrides={"SIGNNOW_TOKEN": "tok"},
        allowed_tools=["a", "b"],
        include_resources=True,
        require_in_path=True,
    )
    # McpToolSpec should be constructed with these values
    kwargs = mock_mcp_spec.call_args.kwargs
    assert kwargs["allowed_tools"] == ["a", "b"]
    assert kwargs["include_resources"] is True
