from pathlib import Path
from unittest import mock

import pytest

from scripts.run_ci_tests import (
    ResultStatus,
    find_integrations,
    find_packs,
    find_utils,
    get_changed_files,
    get_changed_packages,
    get_dependants_packages,
    run_pytest,
)


@pytest.fixture
def base_ref():
    return "main"


@pytest.fixture
def data_path():
    return str(Path(__file__).parent / "data")


@mock.patch("subprocess.run")
def test_get_changed_files(mock_run, tmp_path, base_ref):
    # Mock the subprocess.run result
    mock_process = mock.MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "file1.py\nfile2.py\n\nfile3.py"
    mock_run.return_value = mock_process

    result = get_changed_files(base_ref, tmp_path)

    # Verify subprocess.run was called correctly
    mock_run.assert_called_once_with(
        ["git", "diff", "--name-only", "main...HEAD"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )

    # Verify the result
    assert result == ["file1.py", "file2.py", "file3.py"]


@mock.patch("subprocess.run")
def test_get_changed_files_error(mock_run, tmp_path, base_ref):
    # Mock the subprocess.run failure
    mock_process = mock.MagicMock()
    mock_process.returncode = 1
    mock_process.stderr = "Error in git command"
    mock_run.return_value = mock_process

    with pytest.raises(RuntimeError):
        get_changed_files(base_ref, tmp_path)


def test_get_changed_packages():
    # Setup test data
    changed_files = [
        "llama-index-core/file1.py",
        "llama-index-integrations/vector_stores/pkg1/file2.py",
        "some/other/path/file3.py",
    ]
    all_packages = [
        "llama-index-core",
        "llama-index-integrations/vector_stores/pkg1",
        "llama-index-packs/pkg2",
    ]

    # Call the function
    result = get_changed_packages(changed_files, all_packages)

    # Verify the result
    expected = {"llama-index-core", "llama-index-integrations/vector_stores/pkg1"}
    assert result == expected


@mock.patch("scripts.run_ci_tests.load_pyproject")
def test_get_dependants_packages(mock_load_pyproject):
    # Setup test data
    changed_packages = {
        "llama-index-core",
        "llama-index-integrations/vector_stores/pkg1",
    }
    all_packages = [
        "llama-index-core",
        "llama-index-integrations/vector_stores/pkg1",
        "llama-index-packs/pkg2",
        "llama-index-integrations/llm/pkg3",
    ]

    # Setup mock package names and dependencies
    pkg_data = {
        "llama-index-core": {
            "project": {"name": "llama-index-core", "dependencies": ["bar<2.0"]}
        },
        "llama-index-integrations/vector_stores/pkg1": {
            "project": {
                "name": "llama-index-integrations-vector-stores-pkg1",
                "dependencies": ["foo==1.0.0"],
            }
        },
        "llama-index-packs/pkg2": {
            "project": {
                "name": "pkg2",
                "dependencies": ["llama-index-core==0.8.0", "numpy<1.20.0"],
            }
        },
        "llama-index-integrations/llm/pkg3": {
            "project": {
                "name": "pkg3",
                "dependencies": [
                    "llama-index-integrations-vector-stores-pkg1>0.1.0",
                    "pandas>=1.3.0",
                ],
            }
        },
    }

    # Mock tomli.load to return appropriate project data
    def mock_pkg_data(file_path):
        for pkg_path, data in pkg_data.items():
            if pkg_path in str(file_path):
                return data
        return {"name": "", "deps": []}

    mock_load_pyproject.side_effect = mock_pkg_data

    result = get_dependants_packages(changed_packages, all_packages)
    assert result == {"llama-index-packs/pkg2", "llama-index-integrations/llm/pkg3"}


def test_find_integrations(data_path):
    assert {Path(p).name for p in find_integrations(data_path)} == {"pkg1", "pkg2"}


def test_find_packs(data_path):
    assert {Path(p).name for p in find_packs(data_path)} == {"pack1", "pack2"}


def test_find_utils(data_path):
    assert {Path(p).name for p in find_utils(data_path)} == {"util"}


@mock.patch("scripts.run_ci_tests.load_pyproject")
@mock.patch("subprocess.run")
@mock.patch("time.perf_counter")
def test_run_pytest_success(mock_time, mock_run, mock_load_pyproject):
    # Mock time.perf_counter to return fixed values
    mock_time.side_effect = [0, 10]

    # Mock subprocess.run to return success for both commands
    mock_process1 = mock.MagicMock()
    mock_process1.returncode = 0
    mock_process2 = mock.MagicMock()
    mock_process2.returncode = 0
    mock_process2.stdout = "Tests passed"
    mock_process2.stderr = ""
    mock_run.side_effect = [mock_process1, mock_process2]

    # Call the function
    result = run_pytest("", "test_package", set(), "")

    # Verify the result
    expected = {
        "package": "test_package",
        "status": ResultStatus.TESTS_PASSED,
        "stdout": "Tests passed",
        "stderr": "",
        "time": "10.00s",
    }
    assert result == expected


@mock.patch("scripts.run_ci_tests.load_pyproject")
@mock.patch("subprocess.run")
@mock.patch("time.perf_counter")
def test_run_pytest_install_failure(mock_time, mock_run, mock_load_pyproject):
    # Mock time.perf_counter to return fixed values
    mock_time.side_effect = [0, 5]

    # Mock subprocess.run to return failure for install
    mock_process = mock.MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = ""
    mock_process.stderr = "Install failed"
    mock_run.return_value = mock_process

    # Call the function
    result = run_pytest("", "test_package", set(), "")

    # Verify the result
    expected = {
        "package": "test_package",
        "status": ResultStatus.INSTALL_FAILED,
        "stdout": "",
        "stderr": "Install failed",
        "time": "5.00s",
    }
    assert result == expected


@mock.patch("scripts.run_ci_tests.load_pyproject")
@mock.patch("subprocess.run")
@mock.patch("time.perf_counter")
def test_run_pytest_test_failure(mock_time, mock_run, mock_load_pyproject):
    # Mock time.perf_counter to return fixed values
    mock_time.side_effect = [0, 15]

    # Mock subprocess.run to return success for install but failure for tests
    mock_process1 = mock.MagicMock()
    mock_process1.returncode = 0
    mock_process2 = mock.MagicMock()
    mock_process2.returncode = 1
    mock_process2.stdout = "Test output"
    mock_process2.stderr = "Test failed"
    mock_run.side_effect = [mock_process1, mock_process2]

    # Call the function
    result = run_pytest("", "test_package", set(), "")

    # Verify the result
    expected = {
        "package": "test_package",
        "status": ResultStatus.TESTS_FAILED,
        "stdout": "Test output",
        "stderr": "Test failed",
        "time": "15.00s",
    }
    assert result == expected
