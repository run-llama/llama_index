import json
from datetime import date
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
from llama_dev.cli import cli
from llama_dev.release.changelog import (
    CHANGELOG_PLACEHOLDER,
    _extract_pr_data,
    _get_changelog_text,
    _get_latest_tag,
    _get_pr_numbers,
    _run_command,
    _update_changelog_file,
)


@mock.patch("llama_dev.release.changelog._run_command")
@mock.patch("llama_dev.release.changelog._get_latest_tag")
@mock.patch("llama_dev.release.changelog._get_pr_numbers")
@mock.patch("llama_dev.release.changelog._extract_pr_data")
@mock.patch("llama_dev.release.changelog._get_changelog_text")
@mock.patch("llama_dev.release.changelog._update_changelog_file")
def test_command_nothing_changed(
    mock_update_changelog_file,
    mock_get_changelog_text,
    mock_extract_pr_data,
    mock_get_pr_numbers,
    mock_get_latest_tag,
    mock_run_command,
    data_path,
):
    mock_get_pr_numbers.return_value = []

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--repo-root", data_path, "release", "changelog"],
    )
    assert result.exit_code == 1
    assert "No pull requests found since the last tag" in result.stdout


@mock.patch("llama_dev.release.changelog._run_command")
@mock.patch("llama_dev.release.changelog._get_latest_tag")
@mock.patch("llama_dev.release.changelog._get_pr_numbers")
@mock.patch("llama_dev.release.changelog._extract_pr_data")
@mock.patch("llama_dev.release.changelog._get_changelog_text")
@mock.patch("llama_dev.release.changelog._update_changelog_file")
def test_command_pr_fetch_failed(
    mock_update_changelog_file,
    mock_get_changelog_text,
    mock_extract_pr_data,
    mock_get_pr_numbers,
    mock_get_latest_tag,
    mock_run_command,
    data_path,
):
    mock_get_pr_numbers.return_value = ["42"]
    mock_extract_pr_data.side_effect = FileNotFoundError()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--repo-root", data_path, "release", "changelog"],
    )
    assert result.exit_code == 0
    assert "Could not fetch details for PR #42" in result.stdout


@mock.patch("llama_dev.release.changelog._run_command")
@mock.patch("llama_dev.release.changelog._get_latest_tag")
@mock.patch("llama_dev.release.changelog._get_pr_numbers")
@mock.patch("llama_dev.release.changelog._extract_pr_data")
@mock.patch("llama_dev.release.changelog._get_changelog_text")
@mock.patch("llama_dev.release.changelog._update_changelog_file")
def test_command_dry_run(
    mock_update_changelog_file,
    mock_get_changelog_text,
    mock_extract_pr_data,
    mock_get_pr_numbers,
    mock_get_latest_tag,
    mock_run_command,
    data_path,
):
    mock_get_changelog_text.return_value = "Fake Changelog Here"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--repo-root", data_path, "release", "changelog", "--dry-run"],
    )
    assert result.exit_code == 0
    mock_update_changelog_file.assert_not_called()
    assert "Fake Changelog Here" in result.stdout


@mock.patch("llama_dev.release.changelog._run_command")
@mock.patch("llama_dev.release.changelog._get_latest_tag")
@mock.patch("llama_dev.release.changelog._get_pr_numbers")
@mock.patch("llama_dev.release.changelog._extract_pr_data")
@mock.patch("llama_dev.release.changelog._get_changelog_text")
@mock.patch("llama_dev.release.changelog._update_changelog_file")
def test_command_success(
    mock_update_changelog_file,
    mock_get_changelog_text,
    mock_extract_pr_data,
    mock_get_pr_numbers,
    mock_get_latest_tag,
    mock_run_command,
    data_path,
):
    mock_get_pr_numbers.return_value = ["42"]
    mock_extract_pr_data.return_value = ({"foo": {}}, {"foo": "v1.2.3"})

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--repo-root", data_path, "release", "changelog"],
    )
    assert result.exit_code == 0
    mock_update_changelog_file.assert_called_once()


def test_run_command_success():
    with mock.patch("subprocess.run") as mock_run:
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_run.return_value = mock_result
        assert _run_command("echo 'Success'") == "Success"


def test_run_command_failure():
    with mock.patch("subprocess.run") as mock_run:
        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error"
        mock_run.return_value = mock_result
        with pytest.raises(RuntimeError):
            _run_command("false")


@mock.patch("llama_dev.release.changelog._run_command")
def test_get_latest_tag(mock_run_command):
    mock_run_command.return_value = "v1.2.3"
    assert _get_latest_tag() == "v1.2.3"
    mock_run_command.assert_called_once_with(
        'git describe --tags --match "v[0-9]*" --abbrev=0'
    )


@mock.patch("llama_dev.release.changelog._run_command")
def test_get_pr_numbers(mock_run_command):
    log_output = """
    commit 123 (HEAD -> main)
    feat: new feature (#123)
    commit 456
    fix: a bug (#456)
    commit 789
    docs: update readme
    """
    mock_run_command.return_value = log_output
    pr_numbers = _get_pr_numbers("v1.2.3")
    assert pr_numbers == {"123", "456"}
    mock_run_command.assert_called_once_with(
        'git log v1.2.3..HEAD --pretty="format:%H %s"'
    )


@mock.patch("llama_dev.release.changelog._run_command")
@mock.patch("llama_dev.release.changelog.load_pyproject")
@mock.patch("llama_dev.release.changelog.get_changed_packages")
def test_extract_pr_data(
    mock_get_changed_packages, mock_load_pyproject, mock_run_command
):
    repo_root = Path("/path/to/repo")
    all_packages = [repo_root / "pkg1", repo_root / "pkg2"]
    pr_number = "123"
    pr_json = {
        "number": 123,
        "title": "Test PR",
        "url": "https://github.com/test/repo/pull/123",
        "files": [{"path": "pkg1/file.py"}],
    }
    mock_run_command.return_value = json.dumps(pr_json)
    mock_get_changed_packages.return_value = [repo_root / "pkg1"]
    mock_load_pyproject.return_value = {"project": {"version": "0.1.0"}}

    package_prs, package_versions = _extract_pr_data(repo_root, all_packages, pr_number)

    assert "pkg1" in package_prs
    assert package_prs["pkg1"][0]["number"] == 123
    assert "pkg1" in package_versions
    assert package_versions["pkg1"] == "0.1.0"

    mock_run_command.assert_called_once_with(
        f"gh pr view {pr_number} --json number,title,url,files"
    )
    mock_get_changed_packages.assert_called_once()
    mock_load_pyproject.assert_called_once_with(repo_root / "pkg1")


def test_get_changelog_text():
    package_prs = {
        "pkg1": [
            {
                "number": 123,
                "title": "Feat: New feature",
                "url": "https://github.com/test/repo/pull/123",
            }
        ],
        "pkg2": [
            {
                "number": 456,
                "title": "Fix: A bug",
                "url": "https://github.com/test/repo/pull/456",
            }
        ],
    }
    package_versions = {"pkg1": "0.1.0", "pkg2": "0.2.0"}
    today = date.today().strftime("%Y-%m-%d")
    expected_text = f"""{CHANGELOG_PLACEHOLDER}

## [{today}]

### pkg1 [0.1.0]
- Feat: New feature ([#123](https://github.com/test/repo/pull/123))

### pkg2 [0.2.0]
- Fix: A bug ([#456](https://github.com/test/repo/pull/456))"""

    changelog_text = _get_changelog_text(package_prs, package_versions)
    assert changelog_text == expected_text


def test_update_changelog_file():
    repo_root = Path("/path/to/repo")
    changelog_text = "New changelog content"
    initial_content = (
        f"Some initial content\n{CHANGELOG_PLACEHOLDER}\nSome other content"
    )
    expected_content = f"Some initial content\n{changelog_text}\nSome other content"

    m = mock.mock_open(read_data=initial_content)
    with mock.patch("builtins.open", m):
        _update_changelog_file(repo_root, changelog_text)

    m.assert_called_once_with(repo_root / "CHANGELOG.md", "r+")
    handle = m()
    handle.read.assert_called_once()
    handle.seek.assert_called_once_with(0)
    handle.truncate.assert_called_once()
    handle.write.assert_called_once_with(expected_content)


def test_run_command_no_shell_and_args_list():
    with mock.patch("subprocess.run") as mock_run:
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_run.return_value = mock_result
        _run_command("echo ok")
        args, kwargs = mock_run.call_args
        assert isinstance(args[0], list)
        assert not kwargs.get("shell", False)
