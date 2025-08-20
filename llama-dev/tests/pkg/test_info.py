from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
from llama_dev.cli import cli


@pytest.fixture
def mock_projects():
    return {
        "package1": {"project": {"name": "llama-index-package1", "version": "0.1.0"}},
        "package2": {"project": {"name": "llama-index-package2", "version": "0.2.0"}},
    }


def test_info_with_package_names(mock_projects, data_path):
    runner = CliRunner()

    with (
        mock.patch("llama_dev.pkg.info.is_llama_index_package", return_value=True),
        mock.patch(
            "llama_dev.pkg.info.load_pyproject",
            side_effect=lambda p: mock_projects[p.name],
        ),
        mock.patch("rich.table.Table.add_row") as mock_add_row,
    ):
        result = runner.invoke(
            cli, ["--repo-root", data_path, "pkg", "info", "package1", "package2"]
        )

        assert result.exit_code == 0
        calls = mock_add_row.call_args_list
        assert len(calls) == 2
        assert {c.args for c in calls} == {
            ("llama-index-package2", "0.2.0", str(data_path / "package2")),
            ("llama-index-package1", "0.1.0", str(data_path / "package1")),
        }


def test_info_with_all_flag(mock_projects):
    runner = CliRunner()
    package_paths = [Path("/fake/repo/root/package1"), Path("/fake/repo/root/package2")]

    with (
        mock.patch("llama_dev.pkg.info.find_all_packages", return_value=package_paths),
        mock.patch(
            "llama_dev.pkg.info.load_pyproject",
            side_effect=lambda p: mock_projects[p.name],
        ),
        mock.patch("rich.table.Table.add_row") as mock_add_row,
    ):
        result = runner.invoke(cli, ["pkg", "info", "--all"])

        assert result.exit_code == 0
        assert mock_add_row.call_count == 2
        mock_add_row.assert_any_call(
            "llama-index-package1", "0.1.0", "/fake/repo/root/package1"
        )
        mock_add_row.assert_any_call(
            "llama-index-package2", "0.2.0", "/fake/repo/root/package2"
        )


def test_info_with_args_error():
    runner = CliRunner()
    result = runner.invoke(cli, ["pkg", "info"])

    assert result.exit_code != 0
    assert "Either specify a package name or use the --all flag" in result.output


def test_info_invalid_package():
    runner = CliRunner()
    with mock.patch("llama_dev.pkg.info.is_llama_index_package", return_value=False):
        result = runner.invoke(cli, ["pkg", "info", "invalid-package"])
        assert result.exit_code != 0
        assert "is not a path to a LlamaIndex package" in result.output
