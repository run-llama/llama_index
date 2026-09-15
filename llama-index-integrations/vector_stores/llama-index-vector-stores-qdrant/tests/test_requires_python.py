import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet


def test_requires_python_allows_3_14():
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())
    requires_python = pyproject["project"]["requires-python"]

    assert SpecifierSet(requires_python).contains("3.14.0"), (
        f"requires-python={requires_python!r} excludes Python 3.14, even "
        "though the package's own dependencies (qdrant-client, grpcio, "
        "llama-index-core) all support it"
    )
