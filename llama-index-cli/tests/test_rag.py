import glob
from unittest import mock


from llama_index.cli import command_line
from llama_index.cli.rag import RagCLI


@mock.patch.object(RagCLI, "handle_cli", return_value="noop")
@mock.patch(
    "sys.argv",
    ["llamaindex-cli", "rag", "--files", *glob.glob("**/*.py", recursive=True)],
)
def test_handle_cli_files(mock_handle_cli) -> None:
    command_line.main()
    mock_handle_cli.assert_called_once()
