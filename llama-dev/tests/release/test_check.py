import json
from unittest import mock

import click
import pytest
from llama_dev.cli import cli
from llama_dev.release.check import (
    _get_current_branch_name,
    _get_version_from_pypi,
    _get_version_from_pyproject,
    check,
)


def test_get_current_branch_name():
    with mock.patch("subprocess.check_output", return_value=b"my-branch\n"):
        assert _get_current_branch_name() == "my-branch"


def test_get_version_from_pyproject(tmp_path):
    core_path = tmp_path / "llama-index-core"
    core_path.mkdir()
    pyproject_content = """
[project]
version = \"1.2.3\"
"""
    (core_path / "pyproject.toml").write_text(pyproject_content)
    assert _get_version_from_pyproject(tmp_path) == "1.2.3"


def test_get_version_from_pypi():
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps(
            {"info": {"version": "1.2.3"}}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        assert _get_version_from_pypi() == "1.2.3"


def test_get_version_from_pypi_error():
    with mock.patch("urllib.request.urlopen", side_effect=Exception("test error")):
        with pytest.raises(click.ClickException):
            _get_version_from_pypi()


@pytest.mark.parametrize(
    (
        "test_id",
        "branch_name",
        "pyproject_version",
        "init_version",
        "pypi_version",
        "should_pass",
        "expected_message",
    ),
    [
        (
            "fail",
            "my-release-branch",
            "0.1.1",
            "0.1.1",
            "0.1.0",
            False,
            [
                "❌ To release 'llama-index' you have to checkout the `main` branch.",
            ],
        ),
        (
            "on_main",
            "main",
            "0.1.1",
            "0.1.1",
            "0.1.0",
            True,
            [
                "✅ You are on the `main` branch.",
                "✅ Version 0.1.1 is newer than the latest on PyPI (0.1.0).",
            ],
        ),
        (
            "not_newer",
            "main",
            "0.1.0",
            "0.1.0",
            "0.1.0",
            False,
            [
                "❌ Version 0.1.0 is not newer than the latest on PyPI (0.1.0).",
            ],
        ),
    ],
)
def test_check_command(
    mock_rich_console,
    test_id,
    branch_name,
    pyproject_version,
    init_version,
    pypi_version,
    should_pass,
    expected_message,
):
    with (
        mock.patch(
            "llama_dev.release.check._get_current_branch_name", return_value=branch_name
        ),
        mock.patch(
            "llama_dev.release.check._get_version_from_pyproject",
            return_value=pyproject_version,
        ),
        mock.patch(
            "llama_dev.release.check._get_version_from_pypi", return_value=pypi_version
        ),
    ):
        ctx = click.Context(cli)
        ctx.obj = {"console": mock_rich_console, "repo_root": ""}

        if should_pass:
            ctx.invoke(check, before_core=True)
            # print messages from console
            for msg in expected_message:
                mock_rich_console.print.assert_any_call(msg)
        else:
            with pytest.raises(SystemExit) as e:
                ctx.invoke(check, before_core=True)
            assert e.type is SystemExit
            assert e.value.code == 1
            for msg in expected_message:
                mock_rich_console.print.assert_any_call(msg, style="error")
