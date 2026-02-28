import pytest
from unittest.mock import patch

from llama_index.tools.aws_bedrock_agentcore import AgentCoreCodeInterpreterToolSpec


class TestAsyncCodeInterpreterSessionMethods:
    """Verify all session async methods delegate via asyncio.to_thread."""

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.execute_code"
    )
    async def test_aexecute_code(self, mock_sync):
        mock_sync.return_value = "Hello, World!"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.aexecute_code(
            code="print('Hello')", language="python", thread_id="t1"
        )
        mock_sync.assert_called_once_with(
            code="print('Hello')",
            language="python",
            clear_context=False,
            thread_id="t1",
        )
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.execute_command"
    )
    async def test_aexecute_command(self, mock_sync):
        mock_sync.return_value = "file1.txt"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.aexecute_command(command="ls", thread_id="t1")
        mock_sync.assert_called_once_with(command="ls", thread_id="t1")
        assert result == "file1.txt"

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.read_files"
    )
    async def test_aread_files(self, mock_sync):
        mock_sync.return_value = "content"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.aread_files(paths=["a.txt"], thread_id="t1")
        mock_sync.assert_called_once_with(paths=["a.txt"], thread_id="t1")
        assert result == "content"

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.list_files"
    )
    async def test_alist_files(self, mock_sync):
        mock_sync.return_value = "file1\nfile2"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.alist_files(directory_path="/tmp", thread_id="t1")
        mock_sync.assert_called_once_with(directory_path="/tmp", thread_id="t1")
        assert result == "file1\nfile2"

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.delete_files"
    )
    async def test_adelete_files(self, mock_sync):
        mock_sync.return_value = "deleted"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.adelete_files(paths=["a.txt"], thread_id="t1")
        mock_sync.assert_called_once_with(paths=["a.txt"], thread_id="t1")
        assert result == "deleted"

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.write_files"
    )
    async def test_awrite_files(self, mock_sync):
        mock_sync.return_value = "written"
        files = [{"path": "a.txt", "text": "hello"}]
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.awrite_files(files=files, thread_id="t1")
        mock_sync.assert_called_once_with(files=files, thread_id="t1")
        assert result == "written"

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.start_command"
    )
    async def test_astart_command(self, mock_sync):
        mock_sync.return_value = "task-123"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.astart_command(command="sleep 10", thread_id="t1")
        mock_sync.assert_called_once_with(command="sleep 10", thread_id="t1")
        assert result == "task-123"

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.get_task"
    )
    async def test_aget_task(self, mock_sync):
        mock_sync.return_value = "RUNNING"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.aget_task(task_id="task-123", thread_id="t1")
        mock_sync.assert_called_once_with(task_id="task-123", thread_id="t1")
        assert result == "RUNNING"

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.stop_task"
    )
    async def test_astop_task(self, mock_sync):
        mock_sync.return_value = "stopped"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.astop_task(task_id="task-123", thread_id="t1")
        mock_sync.assert_called_once_with(task_id="task-123", thread_id="t1")
        assert result == "stopped"


class TestAsyncCodeInterpreterLifecycle:
    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.list_code_interpreters"
    )
    async def test_alist_code_interpreters(self, mock_list):
        """Test alist_code_interpreters delegates to sync list_code_interpreters."""
        mock_list.return_value = (
            "Found 1 code interpreter(s):\n"
            "- my-ci (ID: ci-123, Status: ACTIVE, Type: CUSTOM)"
        )

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.alist_code_interpreters(
            interpreter_type="CUSTOM", max_results=5, thread_id="test-thread"
        )

        mock_list.assert_called_once_with(
            interpreter_type="CUSTOM", max_results=5, thread_id="test-thread"
        )
        assert "Found 1 code interpreter(s)" in result

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.create_code_interpreter"
    )
    async def test_acreate_code_interpreter(self, mock_create):
        """Test acreate_code_interpreter delegates to sync create_code_interpreter."""
        mock_create.return_value = (
            "Code interpreter created (ID: ci-456, Status: CREATING)"
        )

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.acreate_code_interpreter(
            name="test_ci",
            execution_role_arn="arn:aws:iam::123456789012:role/test",
            network_mode="PUBLIC",
            description="test desc",
            thread_id="test-thread",
        )

        mock_create.assert_called_once_with(
            name="test_ci",
            execution_role_arn="arn:aws:iam::123456789012:role/test",
            network_mode="PUBLIC",
            description="test desc",
            subnet_ids=None,
            security_group_ids=None,
            thread_id="test-thread",
        )
        assert "Code interpreter created" in result

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.delete_code_interpreter"
    )
    async def test_adelete_code_interpreter(self, mock_delete):
        """Test adelete_code_interpreter delegates to sync delete_code_interpreter."""
        mock_delete.return_value = (
            "Code interpreter 'ci-456' deleted (Status: DELETING)"
        )

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.adelete_code_interpreter(
            interpreter_id="ci-456", thread_id="test-thread"
        )

        mock_delete.assert_called_once_with(
            interpreter_id="ci-456", thread_id="test-thread"
        )
        assert "deleted" in result

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.get_code_interpreter"
    )
    async def test_aget_code_interpreter(self, mock_get):
        """Test aget_code_interpreter delegates to sync get_code_interpreter."""
        mock_get.return_value = (
            "Code interpreter 'ci-456':\n"
            "  Name: test_ci\n"
            "  Status: ACTIVE\n"
            "  Network mode: PUBLIC"
        )

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.aget_code_interpreter(
            interpreter_id="ci-456", thread_id="test-thread"
        )

        mock_get.assert_called_once_with(
            interpreter_id="ci-456", thread_id="test-thread"
        )
        assert "Code interpreter 'ci-456'" in result
        assert "Status: ACTIVE" in result

    @pytest.mark.asyncio
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base"
        ".AgentCoreCodeInterpreterToolSpec.clear_context"
    )
    async def test_aclear_context(self, mock_clear):
        """Test aclear_context delegates to sync clear_context."""
        mock_clear.return_value = "Python execution context cleared successfully."

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        result = await tool_spec.aclear_context(thread_id="test-thread")

        mock_clear.assert_called_once_with(thread_id="test-thread")
        assert "cleared successfully" in result
