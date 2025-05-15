from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
from llama_dev import test as llama_dev_test
from llama_dev.cli import cli
from llama_dev.test import ResultStatus, _pytest, _run_tests


def mocked_coverage_failed(*args, **kwargs):
    return {
        "package": Path(__file__).parent.parent / "data" / "test_integration",
        "status": ResultStatus.COVERAGE_FAILED,
        "stdout": "",
        "stderr": "Coverage below threshold",
        "time": "0.1s",
    }


def mocked_install_failed(*args, **kwargs):
    return {
        "package": Path(__file__).parent.parent / "data" / "test_integration",
        "status": ResultStatus.INSTALL_FAILED,
        "stdout": "",
        "stderr": "Install failed",
        "time": "0.1s",
    }


def mocked_skip_failed(*args, **kwargs):
    return {
        "package": Path(__file__).parent.parent / "data" / "test_integration",
        "status": ResultStatus.SKIPPED,
        "stdout": "",
        "stderr": "Integration skipped",
        "time": "0.1s",
    }


def mocked_success(*args, **kwargs):
    return {
        "package": Path(__file__).parent.parent / "data" / "test_integration",
        "status": ResultStatus.TESTS_PASSED,
        "stdout": "",
        "stderr": "",
        "time": "0.1s",
    }


@pytest.fixture
def changed_packages():
    return {
        Path("/fake/package/dependency1"),
        Path("/fake/package/dependency2"),
    }


@pytest.fixture
def package_data():
    return {
        "project": {
            "dependencies": ["dependency1", "requests"],
            "requires-python": ">=3.8",
        }
    }


def test_test_command_base_ref():
    runner = CliRunner()

    result = runner.invoke(cli, ["test", "--base-ref"])
    assert result.exit_code != 0
    assert "Error: Option '--base-ref' requires an argument." in result.output

    result = runner.invoke(cli, ["test", "--base-ref="])
    assert result.exit_code != 0
    assert "Error: Option '--base-ref' cannot be empty." in result.output


def test_test_command_requires_base_ref_or_packages():
    runner = CliRunner()

    result = runner.invoke(cli, ["test"])
    assert result.exit_code != 1
    assert (
        "Error: Either pass '--base-ref' or provide at least one package name."
        in result.output
    )


def test_test_command_cov_fail_under_requires_cov():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["test", "--base-ref", "main", "--cov-fail-under", "80"]
    )
    assert result.exit_code != 0
    assert "You have to pass --cov in order to use --cov-fail-under" in result.output


@mock.patch("llama_dev.test.find_all_packages")
@mock.patch("llama_dev.test.get_changed_files")
@mock.patch("llama_dev.test.get_changed_packages")
@mock.patch("llama_dev.test.get_dependants_packages")
@mock.patch("llama_dev.test.concurrent.futures.ProcessPoolExecutor")
def test_workers_parameter(
    mock_pool,
    mock_get_dependants,
    mock_get_changed_packages,
    mock_get_changed_files,
    mock_find_all_packages,
):
    # Setup minimal test data
    mock_find_all_packages.return_value = set()
    mock_get_changed_files.return_value = set()
    mock_get_changed_packages.return_value = set()
    mock_get_dependants.return_value = set()

    runner = CliRunner()
    runner.invoke(cli, ["test", "--base-ref", "main", "--workers", "16"])
    mock_pool.assert_called_once_with(max_workers=16)


@mock.patch("llama_dev.test.find_all_packages")
@mock.patch("llama_dev.test.get_changed_files")
@mock.patch("llama_dev.test.get_changed_packages")
@mock.patch("llama_dev.test.get_dependants_packages")
def test_coverage_failures(
    mock_get_dependants,
    mock_get_changed_packages,
    mock_get_changed_files,
    mock_find_all_packages,
    monkeypatch,
    data_path,
):
    mock_find_all_packages.return_value = {Path("/fake/repo/package1")}
    mock_get_changed_files.return_value = {Path("/fake/repo/package1/file.py")}
    mock_get_changed_packages.return_value = {Path("/fake/repo/package1")}
    mock_get_dependants.return_value = set()

    monkeypatch.setattr(llama_dev_test, "_run_tests", mocked_coverage_failed)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--repo-root",
            data_path,
            "test",
            "--base-ref",
            "main",
            "--cov",
            "--cov-fail-under",
            "40",
        ],
    )

    # Check console output
    assert result.exit_code == 1
    assert "❌ test_integration failed" in result.stdout
    assert "Error:\nCoverage below threshold" in result.stdout


@mock.patch("llama_dev.test.find_all_packages")
@mock.patch("llama_dev.test.get_changed_files")
@mock.patch("llama_dev.test.get_changed_packages")
@mock.patch("llama_dev.test.get_dependants_packages")
def test_install_failures(
    mock_get_dependants,
    mock_get_changed_packages,
    mock_get_changed_files,
    mock_find_all_packages,
    monkeypatch,
    data_path,
):
    mock_find_all_packages.return_value = {Path("/fake/repo/package1")}
    mock_get_changed_files.return_value = {Path("/fake/repo/package1/file.py")}
    mock_get_changed_packages.return_value = {Path("/fake/repo/package1")}
    mock_get_dependants.return_value = set()

    monkeypatch.setattr(llama_dev_test, "_run_tests", mocked_install_failed)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--repo-root", data_path, "test", "--base-ref", "main"],
    )

    # Check console output
    assert result.exit_code == 0
    assert "❗ Unable to build package test_integration" in result.stdout
    assert "Error:\nInstall failed" in result.stdout


@mock.patch("llama_dev.test.find_all_packages")
@mock.patch("llama_dev.test.get_changed_files")
@mock.patch("llama_dev.test.get_changed_packages")
@mock.patch("llama_dev.test.get_dependants_packages")
def test_skip_failures(
    mock_get_dependants,
    mock_get_changed_packages,
    mock_get_changed_files,
    mock_find_all_packages,
    monkeypatch,
    data_path,
):
    mock_find_all_packages.return_value = {Path("/fake/repo/package1")}
    mock_get_changed_files.return_value = {Path("/fake/repo/package1/file.py")}
    mock_get_changed_packages.return_value = {Path("/fake/repo/package1")}
    mock_get_dependants.return_value = set()

    monkeypatch.setattr(llama_dev_test, "_run_tests", mocked_skip_failed)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--repo-root", data_path, "test", "--base-ref", "main"],
    )

    # Check console output
    assert result.exit_code == 0
    assert "⏭️  test_integration skipped" in result.stdout
    assert "Error:\nIntegration skipped" in result.stdout


@mock.patch("llama_dev.test.find_all_packages")
@mock.patch("llama_dev.test.get_changed_files")
@mock.patch("llama_dev.test.get_changed_packages")
@mock.patch("llama_dev.test.get_dependants_packages")
def test_success(
    mock_get_dependants,
    mock_get_changed_packages,
    mock_get_changed_files,
    mock_find_all_packages,
    monkeypatch,
    data_path,
):
    mock_find_all_packages.return_value = {Path("/fake/repo/package1")}
    mock_get_changed_files.return_value = {Path("/fake/repo/package1/file.py")}
    mock_get_changed_packages.return_value = {Path("/fake/repo/package1")}
    mock_get_dependants.return_value = set()

    monkeypatch.setattr(llama_dev_test, "_run_tests", mocked_success)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--repo-root", data_path, "test", "--base-ref", "main"],
    )

    # Check console output
    assert result.exit_code == 0
    assert "✅ test_integration succeeded in 0.1s" in result.stdout


@mock.patch("llama_dev.test.find_all_packages")
@mock.patch("llama_dev.test.get_changed_files")
@mock.patch("llama_dev.test.get_changed_packages")
@mock.patch("llama_dev.test.get_dependants_packages")
def test_package_parameter(
    mock_get_dependants,
    mock_get_changed_packages,
    mock_get_changed_files,
    mock_find_all_packages,
    data_path,
):
    # Setup minimal test data
    mock_find_all_packages.return_value = set()
    mock_get_changed_files.return_value = set()
    mock_get_changed_packages.return_value = set()
    mock_get_dependants.return_value = set()

    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "--repo-root",
            data_path,
            "test",
            "--base-ref",
            "main",
            "package_1",
            "package_2",
        ],
    )
    mock_get_dependants.assert_called_with(
        {data_path / "package_1", data_path / "package_2"}, set()
    )


#
# Tests for the utility methods, we call them directly not through cli execution
#


@mock.patch("llama_dev.pkg.cmd_exec.subprocess.run")
def test__pytest(mock_subprocess):
    mock_subprocess.return_value = mock.Mock(
        returncode=0, stdout="Command output", stderr=""
    )

    _pytest(Path(), {}, cov=False)
    assert mock_subprocess.call_args[0][0] == [
        "uv",
        "run",
        "--no-sync",
        "--",
        "pytest",
        "-q",
        "--disable-warnings",
        "--disable-pytest-warnings",
    ]

    mock_subprocess.reset_mock()
    _pytest(Path(), {}, cov=True)
    assert mock_subprocess.call_args[0][0] == [
        "uv",
        "run",
        "--no-sync",
        "--",
        "pytest",
        "-q",
        "--disable-warnings",
        "--disable-pytest-warnings",
        "--cov=.",
        "--cov-report=xml",
    ]


def test_incompatible_python_version(changed_packages):
    with (
        mock.patch(
            "llama_dev.test.load_pyproject",
            return_value={"project": {"requires-python": ">=3.10"}},
        ),
        mock.patch("llama_dev.test.is_python_version_compatible", return_value=False),
    ):
        result = _run_tests(Path(), changed_packages, "main", False, 0)

        assert result["status"] == ResultStatus.SKIPPED
        assert "Not compatible with Python" in result["stderr"]


def test_install_dependencies_failure(changed_packages, package_data):
    with (
        mock.patch("llama_dev.test.load_pyproject", return_value=package_data),
        mock.patch("llama_dev.test.is_python_version_compatible", return_value=True),
        mock.patch(
            "llama_dev.test._uv_sync",
            return_value=mock.Mock(
                returncode=1, stdout="stdout output", stderr="install error"
            ),
        ),
    ):
        result = _run_tests(Path(), changed_packages, "main", False, 0)

        assert result["status"] == ResultStatus.INSTALL_FAILED
        assert result["stderr"] == "install error"


def test_install_local_packages_failure(changed_packages, package_data):
    with (
        mock.patch("llama_dev.test.load_pyproject", return_value=package_data),
        mock.patch("llama_dev.test.is_python_version_compatible", return_value=True),
        mock.patch(
            "llama_dev.test._uv_sync",
            return_value=mock.Mock(returncode=0),
        ),
        mock.patch(
            "llama_dev.test.get_dep_names", return_value=["dependency1", "dependency2"]
        ),
        mock.patch(
            "llama_dev.test._uv_install_local",
            return_value=mock.Mock(
                returncode=1, stdout="stdout", stderr="local install error"
            ),
        ),
    ):
        result = _run_tests(Path(), changed_packages, "main", False, 0)

        assert result["status"] == ResultStatus.INSTALL_FAILED
        assert result["stderr"] == "local install error"


def test_pytest_failure(changed_packages, package_data):
    with (
        mock.patch("llama_dev.test.load_pyproject", return_value=package_data),
        mock.patch("llama_dev.test.is_python_version_compatible", return_value=True),
        mock.patch(
            "llama_dev.test._uv_sync",
            return_value=mock.Mock(returncode=0),
        ),
        mock.patch("llama_dev.test.get_dep_names", return_value=["dependency1"]),
        mock.patch(
            "llama_dev.test._uv_install_local",
            return_value=mock.Mock(returncode=0),
        ),
        mock.patch(
            "llama_dev.test._pytest",
            return_value=mock.Mock(
                returncode=1, stdout="test output", stderr="test failures"
            ),
        ),
    ):
        result = _run_tests(Path(), changed_packages, "main", False, 0)

        assert result["status"] == ResultStatus.TESTS_FAILED
        assert result["stdout"] == "test output"
        assert result["stderr"] == "test failures"


def test_coverage_failure(changed_packages, package_data):
    with (
        mock.patch("llama_dev.test.load_pyproject", return_value=package_data),
        mock.patch("llama_dev.test.is_python_version_compatible", return_value=True),
        mock.patch(
            "llama_dev.test._uv_sync",
            return_value=mock.Mock(returncode=0),
        ),
        mock.patch("llama_dev.test.get_dep_names", return_value=[]),
        mock.patch(
            "llama_dev.test._pytest",
            return_value=mock.Mock(returncode=0, stdout="tests passed"),
        ),
        mock.patch(
            "llama_dev.test._diff_cover",
            return_value=mock.Mock(
                returncode=1,
                stdout="coverage output",
                stderr="coverage below threshold",
            ),
        ),
    ):
        result = _run_tests(Path(), changed_packages, "main", True, 80)

        assert result["status"] == ResultStatus.COVERAGE_FAILED
        assert result["stderr"] == "coverage below threshold"


def test_successful_run(changed_packages, package_data):
    with (
        mock.patch("llama_dev.test.load_pyproject", return_value=package_data),
        mock.patch("llama_dev.test.is_python_version_compatible", return_value=True),
        mock.patch(
            "llama_dev.test._uv_sync",
            return_value=mock.Mock(returncode=0),
        ),
        mock.patch("llama_dev.test.get_dep_names", return_value=[]),
        mock.patch(
            "llama_dev.test._pytest",
            return_value=mock.Mock(returncode=0, stdout="all tests passed", stderr=""),
        ),
    ):
        result = _run_tests(Path(), changed_packages, "main", False, 0)

        assert result["status"] == ResultStatus.TESTS_PASSED
        assert result["stdout"] == "all tests passed"
        assert "time" in result


def test_successful_run_with_coverage(package_data, changed_packages):
    """Test a successful run with coverage checking."""
    with (
        mock.patch("llama_dev.test.load_pyproject", return_value=package_data),
        mock.patch("llama_dev.test.is_python_version_compatible", return_value=True),
        mock.patch(
            "llama_dev.test._uv_sync",
            return_value=mock.Mock(returncode=0),
        ),
        mock.patch("llama_dev.test.get_dep_names", return_value=[]),
        mock.patch(
            "llama_dev.test._pytest",
            return_value=mock.Mock(returncode=0, stdout="all tests passed", stderr=""),
        ),
        mock.patch(
            "llama_dev.test._diff_cover",
            return_value=mock.Mock(returncode=0, stdout="coverage ok", stderr=""),
        ),
    ):
        result = _run_tests(Path(), changed_packages, "main", True, 80)

        assert result["status"] == ResultStatus.TESTS_PASSED
        assert result["stdout"] == "coverage ok"
        assert "time" in result


def test__trim():
    from llama_dev.test import MAX_CONSOLE_PRINT_LINES, _trim

    # Test with a short message (less than MAX_CONSOLE_PRINT_LINES)
    short_msg = "Line 1\nLine 2\nLine 3"
    assert _trim(False, short_msg) == short_msg
    assert _trim(True, short_msg) == short_msg

    # Test with a long message (more than MAX_CONSOLE_PRINT_LINES)
    long_msg = "\n".join([f"Line {i}" for i in range(1, MAX_CONSOLE_PRINT_LINES + 10)])

    # In non-debug mode, the message should be truncated
    trimmed = _trim(False, long_msg)
    trimmed_lines = trimmed.split("\n")

    # Should have MAX_CONSOLE_PRINT_LINES lines plus the additional "truncated" message line
    assert len(trimmed_lines) == MAX_CONSOLE_PRINT_LINES + 1

    # The first MAX_CONSOLE_PRINT_LINES lines should be from the original message
    for i in range(MAX_CONSOLE_PRINT_LINES):
        assert trimmed_lines[i] == f"Line {i + 1}"

    # The last line should be the truncation message
    assert (
        "<-- llama-dev: output truncated, pass '--debug' to see the full log -->"
        in trimmed_lines[-1]
    )

    # In debug mode, the message should not be truncated
    debug_trimmed = _trim(True, long_msg)
    assert debug_trimmed == long_msg
    assert (
        len(debug_trimmed.split("\n")) == MAX_CONSOLE_PRINT_LINES + 9
    )  # Original number of lines
