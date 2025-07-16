from click.testing import CliRunner
from llama_dev.cli import cli
from unittest import mock
from pathlib import Path


def test_version():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert result.output.startswith("cli, version ")


def test_cli_empty_package_only_skipped_for_no_tests(tmp_path):
    # Find the repo root
    repo_root = Path(__file__).parent.parent.parent / "llama-dev"
    # Temporarily create a directory for testing the empty package output in CLI
    test_pkg = repo_root / "empty_package"
    test_pkg.mkdir(exist_ok=True)

    # Create pyproject.toml
    (test_pkg / "pyproject.toml").write_text("[project]\nrequires-python = '>=3.8'\n")

    try:
        # Mock all the method return values
        with (
            mock.patch("llama_dev.test.find_all_packages", return_value={test_pkg}),
            mock.patch("llama_dev.test.get_changed_files", return_value={}),
            mock.patch("llama_dev.test.get_changed_packages", return_value={test_pkg}),
            mock.patch("llama_dev.test.get_dependants_packages", return_value=set()),
            mock.patch(
                "llama_dev.test.load_pyproject",
                return_value={"project": {"requires-python": ">=3.8"}},
            ),
            mock.patch(
                "llama_dev.test.is_python_version_compatible", return_value=True
            ),
            mock.patch("llama_dev.test.package_has_tests", return_value=False),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["--repo-root", str(repo_root), "test", "empty_package"]
            )

        assert "skipped because they have no tests" in result.output
        assert "skipped due to Python version incompatibility" not in result.output
    finally:
        # Clean up
        for f in test_pkg.iterdir():
            f.unlink()
        test_pkg.rmdir()
