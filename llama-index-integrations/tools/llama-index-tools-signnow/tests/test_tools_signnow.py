from unittest.mock import patch, MagicMock


from llama_index.tools.signnow.base import SignNowMCPToolSpec


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in SignNowMCPToolSpec.__mro__]
    assert "BaseToolSpec" in names_of_base_classes


@patch("shutil.which")
def test_from_env_returns_spec(mock_which: MagicMock) -> None:
    mock_which.return_value = "/usr/local/bin/sn-mcp"

    spec = SignNowMCPToolSpec.from_env(
        env_overrides={
            "SIGNNOW_TOKEN": "dummy",
            "IRRELEVANT": "skip",
        },
        require_in_path=True,
    )

    assert isinstance(spec, SignNowMCPToolSpec)
