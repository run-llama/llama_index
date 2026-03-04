from unittest import mock

from click.testing import CliRunner
from llama_dev.cli import cli


def test_cmd_exec_no_package_no_all_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["pkg", "exec", "--cmd", "echo hello"])
    assert result.exit_code != 0
    assert "Either specify a package name or use the --all flag" in result.output


@mock.patch("llama_dev.pkg.cmd_exec.is_llama_index_package")
def test_cmd_exec_invalid_package(mock_is_llama_index):
    mock_is_llama_index.return_value = False

    runner = CliRunner()
    result = runner.invoke(cli, ["pkg", "exec", "invalid-pkg", "--cmd", "echo hello"])

    assert result.exit_code != 0
    print(result.output)
    assert "not a path to a LlamaIndex package" in result.output


@mock.patch("llama_dev.pkg.cmd_exec.find_all_packages")
@mock.patch("llama_dev.pkg.cmd_exec.subprocess.run")
def test_cmd_exec_all_flag(mock_subprocess, mock_find_all, data_path):
    mock_find_all.return_value = [data_path / "fake/pkg1", data_path / "fake/pkg2"]
    mock_subprocess.return_value = mock.Mock(
        returncode=0, stdout="Command output", stderr=""
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["pkg", "exec", "--all", "--cmd", "echo hello"])

    assert result.exit_code == 0
    assert mock_subprocess.call_count == 2
    assert "Command succeeded" in result.output


@mock.patch("llama_dev.pkg.cmd_exec.is_llama_index_package")
@mock.patch("llama_dev.pkg.cmd_exec.subprocess.run")
def test_cmd_exec_single_package_success(mock_subprocess, mock_is_llama_index):
    mock_is_llama_index.return_value = True
    mock_subprocess.return_value = mock.Mock(
        returncode=0, stdout="Command output", stderr=""
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["pkg", "exec", "valid-pkg", "--cmd", "echo hello"])

    assert result.exit_code == 0
    assert mock_subprocess.call_count == 1
    assert "Command succeeded" in result.output


@mock.patch("llama_dev.pkg.cmd_exec.is_llama_index_package")
@mock.patch("llama_dev.pkg.cmd_exec.subprocess.run")
def test_cmd_exec_failure_without_fail_fast(mock_subprocess, mock_is_llama_index):
    mock_is_llama_index.return_value = True
    mock_subprocess.return_value = mock.Mock(
        returncode=1, stdout="", stderr="Error message"
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["pkg", "exec", "valid-pkg", "--cmd", "echo hello"])

    assert result.exit_code == 0  # Command should continue despite failure
    assert "Command 'echo hello' failed" in result.output


@mock.patch("llama_dev.pkg.cmd_exec.is_llama_index_package")
@mock.patch("llama_dev.pkg.cmd_exec.subprocess.run")
def test_cmd_exec_failure_with_fail_fast(mock_subprocess, mock_is_llama_index):
    mock_is_llama_index.return_value = True
    mock_subprocess.return_value = mock.Mock(
        returncode=1, stdout="", stderr="Error message"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["pkg", "exec", "valid-pkg", "--cmd", "echo hello", "--fail-fast"]
    )

    assert result.exit_code != 0  # Command should fail at the first error
    assert "Command 'echo hello' failed" in result.output


@mock.patch("llama_dev.pkg.cmd_exec.is_llama_index_package")
@mock.patch("llama_dev.pkg.cmd_exec.subprocess.run")
def test_cmd_exec_multiple_packages(mock_subprocess, mock_is_llama_index):
    mock_is_llama_index.return_value = True
    mock_subprocess.return_value = mock.Mock(
        returncode=0, stdout="Command output", stderr=""
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["pkg", "exec", "pkg1", "pkg2", "--cmd", "echo hello"])

    assert result.exit_code == 0
    assert mock_subprocess.call_count == 2
    assert "Command succeeded" in result.output


@mock.patch("llama_dev.pkg.cmd_exec.is_llama_index_package")
@mock.patch("llama_dev.pkg.cmd_exec.subprocess.run")
def test_cmd_exec_silent(mock_subprocess, mock_is_llama_index):
    mock_is_llama_index.return_value = True
    mock_subprocess.return_value = mock.Mock(
        returncode=0, stdout="Command output", stderr=""
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["pkg", "exec", "valid-pkg", "--silent", "--cmd", "echo hello"]
    )

    assert result.exit_code == 0
    assert mock_subprocess.call_count == 1
    assert result.output == ""
