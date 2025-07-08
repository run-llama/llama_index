from pathlib import Path
from unittest import mock

import pytest
from llama_dev.utils import (
    find_all_packages,
    find_integrations,
    find_packs,
    find_utils,
    get_changed_files,
    get_changed_packages,
    get_dep_names,
    get_dependants_packages,
    is_python_version_compatible,
    load_pyproject,
    package_has_tests,
)


def test_find_integrations(data_path):
    assert {p.name for p in find_integrations(data_path)} == {"pkg1", "pkg2"}


def test_find_packs(data_path):
    assert {p.name for p in find_packs(data_path)} == {"pack1", "pack2"}


def test_find_utils(data_path):
    assert {p.name for p in find_utils(data_path)} == {"util"}


def test_package_has_tests(data_path):
    assert not package_has_tests(data_path / "llama-index-packs" / "pack2")


def test_load_pyproject(data_path):
    pkg_data = load_pyproject(
        data_path / "llama-index-integrations" / "vector_stores" / "pkg1"
    )
    assert pkg_data["project"]["name"] == "pkg1"


def test_find_all_packages(data_path):
    assert len(find_all_packages(data_path)) == 8


@mock.patch("subprocess.run")
def test_get_changed_files(mock_run, tmp_path):
    # Mock the subprocess.run result
    mock_process = mock.MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "file1.py\nfile2.py\n\nfile3.py"
    mock_run.return_value = mock_process

    result = get_changed_files(tmp_path, "my-branch")

    assert result == [
        tmp_path / "file1.py",
        tmp_path / "file2.py",
        tmp_path / "file3.py",
    ]
    mock_run.assert_called_once_with(
        ["git", "diff", "--name-only", "my-branch...HEAD"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )


@mock.patch("subprocess.run")
def test_get_changed_files_error(mock_run, tmp_path):
    # Mock the subprocess.run failure
    mock_process = mock.MagicMock()
    mock_process.returncode = 1
    mock_process.stderr = "Error in git command"
    mock_run.return_value = mock_process

    with pytest.raises(RuntimeError):
        get_changed_files(tmp_path)


def test_get_changed_packages():
    changed_files = [
        Path("./llama-index-core/file1.py"),
        Path("./llama-index-integrations/vector_stores/pkg1/file2.py"),
        Path("./some/other/path/file3.py"),
    ]
    all_packages = [
        Path("./llama-index-core"),
        Path("./llama-index-integrations/vector_stores/pkg1"),
        Path("./llama-index-packs/pkg2"),
    ]

    result = get_changed_packages(changed_files, all_packages)
    assert result == {
        Path("llama-index-core"),
        Path("llama-index-integrations/vector_stores/pkg1"),
    }


def test_get_dep_names():
    pyproject_data = {
        "project": {
            "dependencies": [
                "numpy>=1.20.0",
                "pandas==1.5.0",
                "scipy<1.9.0",
                "matplotlib",
                "requests>=2.25.0,<3.0.0",
                "beautifulsoup4>=4.9.3,!=4.10.0",
                "urllib3>=1.26.0,<2.0.0,!=1.26.5",
                "sqlalchemy[postgresql]>=1.4.0",
                "django[bcrypt]",
                "colorama; platform_system=='Windows'",
                "importlib-metadata; python_version<'3.8'",
                " tensorflow >= 2.0.0 ",  # extra spaces
                "pillow  ==9.0.0",  # double spaces
                "package @ git+https://github.com/user/repo.git",
                "local-package @ file:///path/to/package",
                "===",  # Invalid dependency string with only separators
                "",  # Empty string
                " ",  # Just a space
            ]
        }
    }

    assert get_dep_names(pyproject_data) == {
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "requests",
        "beautifulsoup4",
        "urllib3",
        "sqlalchemy",
        "django",
        "colorama",
        "importlib-metadata",
        "tensorflow",
        "pillow",
        "package",
        "local-package",
    }

    # Test with empty dependencies
    pyproject_data = {"project": {"dependencies": []}}
    dependencies = get_dep_names(pyproject_data)
    assert dependencies == set()


def test_is_python_version_compatible(mock_current_version):
    # Test with missing 'project' section in pyproject data."""
    pyproject_data = {}
    assert is_python_version_compatible(pyproject_data) is True

    # Test when no Python version requirement is specified
    pyproject_data = {"project": {}}
    assert is_python_version_compatible(pyproject_data) is True

    # Test when the current Python version exactly matches the requirement
    mock_current_version(3, 8, 0)
    pyproject_data = {"project": {"requires-python": "==3.8.0"}}
    assert is_python_version_compatible(pyproject_data) is True

    # Test when the current Python version is within a specified range
    mock_current_version(3, 9, 5)
    pyproject_data = {"project": {"requires-python": ">=3.8,<3.11"}}
    assert is_python_version_compatible(pyproject_data) is True

    # Test when the current Python version is incompatible with requirements
    mock_current_version(3, 7, 0)
    pyproject_data = {"project": {"requires-python": ">=3.8"}}
    assert is_python_version_compatible(pyproject_data) is False

    # Test with a complex version specifier
    mock_current_version(3, 10, 2)
    pyproject_data = {"project": {"requires-python": ">=3.8,!=3.9.0,<3.11"}}
    assert is_python_version_compatible(pyproject_data) is True

    # Test with an invalid version specifier
    pyproject_data = {"project": {"requires-python": "invalid-specifier"}}
    assert is_python_version_compatible(pyproject_data) is True


@mock.patch("llama_dev.utils.load_pyproject")
def test_get_dependants_packages(mock_load_pyproject):
    # Setup test data
    changed_packages = {
        Path("llama-index-core"),
        Path("llama-index-integrations/vector_stores/pkg1"),
    }
    all_packages = [
        Path("llama-index-core"),
        Path("llama-index-integrations/vector_stores/pkg1"),
        Path("llama-index-packs/pkg2"),
        Path("llama-index-integrations/llm/pkg3"),
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
    assert result == {
        Path("llama-index-packs/pkg2"),
        Path("llama-index-integrations/llm/pkg3"),
    }
