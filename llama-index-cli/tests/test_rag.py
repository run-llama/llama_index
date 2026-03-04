import glob
import subprocess
from pathlib import Path
from unittest import mock
import tempfile

from llama_index.cli import command_line
from llama_index.cli.rag import RagCLI
from llama_index.core.ingestion import IngestionPipeline


@mock.patch.object(RagCLI, "handle_cli", return_value="noop")
@mock.patch(
    "sys.argv",
    ["llamaindex-cli", "rag", "--files", *glob.glob("**/*.py", recursive=True)],
)
def test_handle_cli_files(mock_handle_cli) -> None:
    command_line.main()
    mock_handle_cli.assert_called_once()


@mock.patch("os.path.exists")
@mock.patch("llama_index.cli.rag.base.subprocess.run")
@mock.patch("shutil.which")
def test_create_llama_subprocess_call_security(
    mock_which, mock_subprocess_run, mock_exists
):
    """Test that subprocess.run is called with shell=False for security."""
    # Setup
    mock_which.return_value = "/usr/local/bin/npx"  # npx is available
    mock_subprocess_run.return_value = mock.Mock(returncode=0)

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock os.path.exists to return True for test path, False for persist_dir check
        def exists_side_effect(path):
            if path == "/test/data/folder":
                return True  # Test path exists
            elif path == temp_dir:
                return False  # persist_dir doesn't exist (for clear check)
            return str(path).endswith("files_history.txt")

        mock_exists.side_effect = exists_side_effect

        # Mock the IngestionPipeline to avoid OpenAI API key requirement
        with mock.patch(
            "llama_index.core.ingestion.IngestionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_instance = mock.Mock(spec=IngestionPipeline)
            mock_pipeline_instance.vector_store = None
            mock_pipeline_class.return_value = mock_pipeline_instance

            # Create a RagCLI instance with test configuration
            with mock.patch(
                "llama_index.cli.rag.base._try_load_openai_llm"
            ) as mock_llm:
                mock_llm.return_value = mock.Mock()
                rag_cli = RagCLI(
                    persist_dir=temp_dir, ingestion_pipeline=mock_pipeline_instance
                )

        # Create the history file with a single path
        history_file = Path(temp_dir) / "files_history.txt"
        test_path = "/test/data/folder"
        history_file.write_text(test_path + "\n")

        # Execute the code path that calls subprocess.run
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag_cli.handle_cli(create_llama=True))
        loop.close()

        # Verify subprocess.run was called with the correct arguments
        mock_subprocess_run.assert_called_once()

        # Get the call arguments
        call_args = mock_subprocess_run.call_args

        # Verify shell=False is used (check=True is passed, shell is not passed or is False)
        assert call_args[1].get("shell", False) is False, (
            "subprocess.run must not use shell=True"
        )

        # Verify check=True is set for proper error handling
        assert call_args[1].get("check") is True, "subprocess.run should use check=True"

        # Verify the command is passed as a list (required for shell=False)
        command = call_args[0][0]
        assert isinstance(command, list), "Command must be a list when shell=False"

        # Verify the command structure
        expected_command = [
            "npx",
            "create-llama@latest",
            "--frontend",
            "--template",
            "streaming",
            "--framework",
            "fastapi",
            "--ui",
            "shadcn",
            "--vector-db",
            "none",
            "--engine",
            "context",
            "--files",
            test_path,
        ]
        assert command == expected_command, f"Command mismatch: {command}"


@mock.patch("os.path.exists")
@mock.patch("llama_index.cli.rag.base.subprocess.run")
@mock.patch("shutil.which")
def test_create_llama_handles_special_characters_in_path(
    mock_which, mock_subprocess_run, mock_exists
):
    """Test that paths with special characters are handled safely."""
    # Setup
    mock_which.return_value = "/usr/local/bin/npx"
    mock_subprocess_run.return_value = mock.Mock(returncode=0)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with a path containing spaces and special characters
        test_path = "/test/data folder/with spaces & special; chars"

        # Mock os.path.exists to return True for test path
        def exists_side_effect(path):
            if path == test_path:
                return True
            return str(path).endswith("files_history.txt")

        mock_exists.side_effect = exists_side_effect

        # Mock the IngestionPipeline to avoid OpenAI API key requirement
        with mock.patch(
            "llama_index.core.ingestion.IngestionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_instance = mock.Mock(spec=IngestionPipeline)
            mock_pipeline_instance.vector_store = None
            mock_pipeline_class.return_value = mock_pipeline_instance

            # Create a RagCLI instance with test configuration
            with mock.patch(
                "llama_index.cli.rag.base._try_load_openai_llm"
            ) as mock_llm:
                mock_llm.return_value = mock.Mock()
                rag_cli = RagCLI(
                    persist_dir=temp_dir, ingestion_pipeline=mock_pipeline_instance
                )

        # Create history file
        history_file = Path(temp_dir) / "files_history.txt"
        history_file.write_text(test_path + "\n")

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag_cli.handle_cli(create_llama=True))
        loop.close()

        # Verify the path is passed as-is in the list (no shell escaping needed)
        call_args = mock_subprocess_run.call_args
        command = call_args[0][0]
        assert command[-1] == test_path, "Path should be passed unchanged in list form"
        assert call_args[1].get("shell", False) is False, "Must not use shell=True"


@mock.patch("os.path.exists")
@mock.patch("llama_index.cli.rag.base.subprocess.run")
@mock.patch("shutil.which")
def test_create_llama_subprocess_error_handling(
    mock_which, mock_subprocess_run, mock_exists
):
    """Test that subprocess errors are properly handled."""
    # Setup
    mock_which.return_value = "/usr/local/bin/npx"

    # Simulate subprocess failure
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, ["npx"])

    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = "/test/data/folder"

        # Mock os.path.exists to return True for test path
        def exists_side_effect(path):
            if path == test_path:
                return True
            return str(path).endswith("files_history.txt")

        mock_exists.side_effect = exists_side_effect

        # Mock the IngestionPipeline to avoid OpenAI API key requirement
        with mock.patch(
            "llama_index.core.ingestion.IngestionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_instance = mock.Mock(spec=IngestionPipeline)
            mock_pipeline_instance.vector_store = None
            mock_pipeline_class.return_value = mock_pipeline_instance

            # Create a RagCLI instance with test configuration
            with mock.patch(
                "llama_index.cli.rag.base._try_load_openai_llm"
            ) as mock_llm:
                mock_llm.return_value = mock.Mock()
                rag_cli = RagCLI(
                    persist_dir=temp_dir, ingestion_pipeline=mock_pipeline_instance
                )

        history_file = Path(temp_dir) / "files_history.txt"
        history_file.write_text(test_path + "\n")

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Should raise CalledProcessError due to check=True
        try:
            loop.run_until_complete(rag_cli.handle_cli(create_llama=True))
            raise AssertionError("Should have raised CalledProcessError")
        except subprocess.CalledProcessError:
            pass  # Expected
        finally:
            loop.close()

        # Verify subprocess.run was called with check=True
        call_args = mock_subprocess_run.call_args
        assert call_args[1].get("check") is True, "subprocess.run should use check=True"
