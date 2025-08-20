from unittest.mock import patch, MagicMock
import os
import json

from llama_index.tools.aws_bedrock_agentcore import AgentCoreCodeInterpreterToolSpec
from llama_index.tools.aws_bedrock_agentcore.code_interpreter.base import (
    extract_output_from_stream,
    get_aws_region,
)


class TestGetAwsRegion:
    @patch.dict(os.environ, {"AWS_REGION": "us-east-1"})
    def test_get_aws_region_from_aws_region(self):
        assert get_aws_region() == "us-east-1"

    @patch.dict(
        os.environ, {"AWS_DEFAULT_REGION": "us-west-1", "AWS_REGION": ""}, clear=True
    )
    def test_get_aws_region_from_aws_default_region(self):
        assert get_aws_region() == "us-west-1"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_aws_region_default(self):
        assert get_aws_region() == "us-west-2"


class TestExtractOutputFromStream:
    def test_extract_output_text_only(self):
        response = {
            "stream": [
                {
                    "result": {
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": " World"},
                        ]
                    }
                }
            ]
        }

        output = extract_output_from_stream(response)
        assert output == "Hello\n World"

    def test_extract_output_with_resource_text(self):
        response = {
            "stream": [
                {
                    "result": {
                        "content": [
                            {"type": "text", "text": "Created file:"},
                            {
                                "type": "resource",
                                "resource": {
                                    "uri": "file:///tmp/test.py",
                                    "text": "print('Hello World')",
                                },
                            },
                        ]
                    }
                }
            ]
        }

        output = extract_output_from_stream(response)
        assert (
            output
            == "Created file:\n==== File: /tmp/test.py ====\nprint('Hello World')\n"
        )

    def test_extract_output_with_resource_no_text(self):
        resource_data = {"uri": "file:///tmp/image.png", "mime": "image/png"}
        response = {
            "stream": [
                {
                    "result": {
                        "content": [
                            {"type": "text", "text": "Generated image:"},
                            {"type": "resource", "resource": resource_data},
                        ]
                    }
                }
            ]
        }

        output = extract_output_from_stream(response)
        assert output == f"Generated image:\n{json.dumps(resource_data)}"

    def test_extract_output_multiple_events(self):
        response = {
            "stream": [
                {"result": {"content": [{"type": "text", "text": "First part"}]}},
                {"result": {"content": [{"type": "text", "text": "Second part"}]}},
            ]
        }

        output = extract_output_from_stream(response)
        assert output == "First part\nSecond part"


class TestAgentCoreCodeInterpreterToolSpec:
    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.CodeInterpreter"
    )
    def test_init(self, mock_code_interpreter):
        tool_spec = AgentCoreCodeInterpreterToolSpec(region="us-east-1")
        assert tool_spec.region == "us-east-1"
        assert tool_spec._code_interpreters == {}

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.get_aws_region"
    )
    def test_init_default_region(self, mock_get_aws_region):
        mock_get_aws_region.return_value = "us-west-2"
        tool_spec = AgentCoreCodeInterpreterToolSpec()
        assert tool_spec.region == "us-west-2"
        mock_get_aws_region.assert_called_once()

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.CodeInterpreter"
    )
    def test_get_or_create_interpreter_new(self, mock_code_interpreter):
        mock_instance = MagicMock()
        mock_code_interpreter.return_value = mock_instance

        tool_spec = AgentCoreCodeInterpreterToolSpec(region="us-east-1")
        interpreter = tool_spec._get_or_create_interpreter("test-thread")

        assert interpreter == mock_instance
        assert "test-thread" in tool_spec._code_interpreters
        assert tool_spec._code_interpreters["test-thread"] == mock_instance

        mock_code_interpreter.assert_called_once_with(region="us-east-1")
        mock_instance.start.assert_called_once()

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.CodeInterpreter"
    )
    def test_get_or_create_interpreter_existing(self, mock_code_interpreter):
        mock_instance = MagicMock()

        tool_spec = AgentCoreCodeInterpreterToolSpec(region="us-east-1")
        tool_spec._code_interpreters["test-thread"] = mock_instance

        interpreter = tool_spec._get_or_create_interpreter("test-thread")

        assert interpreter == mock_instance
        mock_code_interpreter.assert_not_called()
        mock_instance.start.assert_not_called()

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_execute_code(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {"result": {"content": [{"type": "text", "text": "Hello World"}]}}
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "Hello World"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.execute_code(
            code="print('Hello World')",
            language="python",
            clear_context=True,
            thread_id="test-thread",
        )

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="executeCode",
            params={
                "code": "print('Hello World')",
                "language": "python",
                "clearContext": True,
            },
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "Hello World"

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_execute_code_exception(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.invoke.side_effect = Exception("Test error")

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.execute_code(
            code="print('Hello World')", language="python", thread_id="test-thread"
        )

        assert "Error executing code: Test error" in result
        mock_extract_output.assert_not_called()

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_execute_command(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {"result": {"content": [{"type": "text", "text": "command output"}]}}
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "command output"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.execute_command(command="ls -la", thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="executeCommand", params={"command": "ls -la"}
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "command output"

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_read_files(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {"result": {"content": [{"type": "text", "text": "file content"}]}}
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "file content"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.read_files(paths=["/tmp/test.txt"], thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="readFiles", params={"paths": ["/tmp/test.txt"]}
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "file content"

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_list_files(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {
                    "result": {
                        "content": [{"type": "text", "text": "file1.txt\nfile2.txt"}]
                    }
                }
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "file1.txt\nfile2.txt"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.list_files(directory_path="/tmp", thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="listFiles", params={"directoryPath": "/tmp"}
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "file1.txt\nfile2.txt"

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_delete_files(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {"result": {"content": [{"type": "text", "text": "Files deleted"}]}}
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "Files deleted"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.delete_files(
            paths=["/tmp/test.txt"], thread_id="test-thread"
        )

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="removeFiles", params={"paths": ["/tmp/test.txt"]}
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "Files deleted"

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_write_files(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {"result": {"content": [{"type": "text", "text": "Files written"}]}}
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "Files written"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        files = [{"path": "/tmp/test.txt", "text": "Hello World"}]
        result = tool_spec.write_files(files=files, thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="writeFiles", params={"content": files}
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "Files written"

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_start_command(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {
                    "result": {
                        "content": [{"type": "text", "text": "Task started: task-123"}]
                    }
                }
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "Task started: task-123"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.start_command(command="sleep 10", thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="startCommandExecution", params={"command": "sleep 10"}
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "Task started: task-123"

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_get_task(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {
                    "result": {
                        "content": [{"type": "text", "text": "Task status: running"}]
                    }
                }
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "Task status: running"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.get_task(task_id="task-123", thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="getTask", params={"taskId": "task-123"}
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "Task status: running"

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.extract_output_from_stream"
    )
    def test_stop_task(self, mock_extract_output):
        mock_code_interpreter = MagicMock()
        mock_response = {
            "stream": [
                {"result": {"content": [{"type": "text", "text": "Task stopped"}]}}
            ]
        }
        mock_code_interpreter.invoke.return_value = mock_response
        mock_extract_output.return_value = "Task stopped"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.stop_task(task_id="task-123", thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.invoke.assert_called_once_with(
            method="stopTask", params={"taskId": "task-123"}
        )
        mock_extract_output.assert_called_once_with(mock_response)
        assert result == "Task stopped"
