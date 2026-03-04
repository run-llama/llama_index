import builtins
from unittest import mock

from click.testing import CliRunner
from llama_dev.cli import cli
from llama_dev.pkg.bump import BumpType, bump_version, update_pyproject_version


def test_bump_version():
    assert bump_version("0.0", BumpType.PATCH) == "0.0.1"
    assert bump_version("0.0", BumpType.MINOR) == "0.1.0"
    assert bump_version("1.0.0", BumpType.MAJOR) == "2.0.0"


def test_update_pyproject_version(data_path):
    pkg = data_path / "llama-index-utils/util"

    # Use the true open if it's the first call
    real_open = open
    mocked_open = mock.mock_open(read_data='[project]\nversion = "1.0.0"\n')
    mocked_open.side_effect = (
        lambda *args, **kwargs: real_open(*args, **kwargs)
        if mocked_open.call_count == 0
        else mock.DEFAULT
    )

    with mock.patch.object(builtins, "open", mocked_open):
        update_pyproject_version(pkg, "99.0.0")

    # Ensure open is called in both read and write modes
    mocked_open.assert_any_call(pkg / "pyproject.toml", "r")
    mocked_open.assert_any_call(pkg / "pyproject.toml", "w")
    # Expect the version to be updated in the string that's attempted to be written
    assert 'version = "99.0.0"' in mocked_open().write.call_args[0][0]


def test_bump_command_no_arguments():
    runner = CliRunner()
    result = runner.invoke(cli, ["pkg", "bump"])
    assert result.exit_code != 0
    assert "Either specify package name(s) or use the --all flag" in result.output


def test_bump_command_with_all_flag():
    runner = CliRunner()

    with (
        mock.patch(
            "llama_dev.pkg.bump.find_all_packages",
            return_value={"package1", "package2"},
        ),
        mock.patch(
            "llama_dev.pkg.bump.load_pyproject",
            return_value={"project": {"version": "1.0.0", "name": "package1"}},
        ),
        mock.patch(
            "llama_dev.pkg.bump.update_pyproject_version"
        ) as mock_update_version,
    ):
        result = runner.invoke(cli, ["pkg", "bump", "--all"])
        assert result.exit_code == 0
        assert mock_update_version.call_count == 2


def test_bump_command_specific_packages(data_path):
    runner = CliRunner()

    with (
        mock.patch(
            "llama_dev.pkg.bump.load_pyproject",
            return_value={"project": {"version": "1.0.0", "name": "package1"}},
        ),
        mock.patch("llama_dev.pkg.bump.is_llama_index_package", return_value=True),
        mock.patch(
            "llama_dev.pkg.bump.update_pyproject_version"
        ) as mock_update_version,
    ):
        result = runner.invoke(cli, ["pkg", "bump", f"{data_path / 'package1'}"])
        assert result.exit_code == 0
        mock_update_version.assert_called_once_with(data_path / "package1", "1.0.1")


def test_bump_command_specific_packages_dry_run(data_path):
    runner = CliRunner()

    with (
        mock.patch(
            "llama_dev.pkg.bump.load_pyproject",
            return_value={"project": {"version": "1.0.0", "name": "package1"}},
        ),
        mock.patch("llama_dev.pkg.bump.is_llama_index_package", return_value=True),
        mock.patch(
            "llama_dev.pkg.bump.update_pyproject_version"
        ) as mock_update_version,
    ):
        result = runner.invoke(
            cli, ["pkg", "bump", f"{data_path / 'package1'}", "--dry-run"]
        )
        assert result.exit_code == 0
        mock_update_version.assert_not_called()
        assert "Would bump tests/data/package1 from 1.0.0 to 1.0.1" in result.output


def test_bump_command_specific_packages_not_a_package(data_path):
    runner = CliRunner()

    with (
        mock.patch(
            "llama_dev.pkg.bump.load_pyproject",
            return_value={"project": {"version": "1.0.0", "name": "package1"}},
        ),
        mock.patch("llama_dev.pkg.bump.is_llama_index_package", return_value=False),
        mock.patch(
            "llama_dev.pkg.bump.update_pyproject_version"
        ) as mock_update_version,
    ):
        result = runner.invoke(cli, ["pkg", "bump", f"{data_path / 'package1'}"])
        assert result.exit_code != 0
        mock_update_version.assert_not_called()
        assert "package1 is not a path to a LlamaIndex package" in result.output
