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

        mock_code_interpreter.assert_called_once_with(
            region="us-east-1", integration_source="llamaindex"
        )
        mock_instance.start.assert_called_once_with()

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.code_interpreter.base.CodeInterpreter"
    )
    def test_get_or_create_interpreter_custom_identifier(self, mock_code_interpreter):
        mock_instance = MagicMock()
        mock_code_interpreter.return_value = mock_instance

        tool_spec = AgentCoreCodeInterpreterToolSpec(
            region="us-east-1", identifier="my-custom-id"
        )
        interpreter = tool_spec._get_or_create_interpreter("test-thread")

        assert interpreter == mock_instance
        mock_code_interpreter.assert_called_once_with(
            region="us-east-1", integration_source="llamaindex"
        )
        mock_instance.start.assert_called_once_with(identifier="my-custom-id")

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

    def test_upload_file(self):
        mock_code_interpreter = MagicMock()

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.upload_file(
            path="data.csv",
            content="a,b\n1,2",
            description="CSV test data",
            thread_id="test-thread",
        )

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.upload_file.assert_called_once_with(
            path="data.csv", content="a,b\n1,2", description="CSV test data"
        )
        assert result == "Uploaded file to data.csv"

    def test_upload_file_exception(self):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.upload_file.side_effect = Exception("Upload failed")

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.upload_file(
            path="data.csv", content="a,b\n1,2", thread_id="test-thread"
        )

        assert "Error uploading file: Upload failed" in result

    def test_upload_files(self):
        mock_code_interpreter = MagicMock()

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        files = [
            {"path": "a.txt", "content": "hello"},
            {"path": "b.txt", "content": "world"},
        ]
        result = tool_spec.upload_files(files=files, thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.upload_files.assert_called_once_with(files=files)
        assert result == "Uploaded 2 file(s)"

    def test_install_packages(self):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.install_packages.return_value = {
            "stdout": "Successfully installed pandas-2.0.0",
            "stderr": "",
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.install_packages(
            packages=["pandas>=2.0"], upgrade=True, thread_id="test-thread"
        )

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.install_packages.assert_called_once_with(
            packages=["pandas>=2.0"], upgrade=True
        )
        assert "Successfully installed pandas-2.0.0" in result

    def test_install_packages_exception(self):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.install_packages.side_effect = Exception("pip failed")

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.install_packages(
            packages=["nonexistent"], thread_id="test-thread"
        )

        assert "Error installing packages: pip failed" in result

    def test_download_file_text(self):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.download_file.return_value = "file content here"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.download_file(path="/tmp/test.txt", thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.download_file.assert_called_once_with(
            path="/tmp/test.txt"
        )
        assert result == "file content here"

    def test_download_file_binary(self):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.download_file.return_value = b"\x89PNG\r\n"

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.download_file(path="/tmp/image.png", thread_id="test-thread")

        assert "[base64 encoded binary file: /tmp/image.png]" in result
        assert "iVBORw0K" in result  # base64 of PNG header

    def test_download_file_exception(self):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.download_file.side_effect = Exception("Not found")

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.download_file(path="/tmp/missing", thread_id="test-thread")

        assert "Error downloading file: Not found" in result

    def test_download_files(self):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.download_files.return_value = {
            "/tmp/a.txt": "content a",
            "/tmp/b.bin": b"\x00\x01",
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.download_files(
            paths=["/tmp/a.txt", "/tmp/b.bin"], thread_id="test-thread"
        )

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.download_files.assert_called_once_with(
            paths=["/tmp/a.txt", "/tmp/b.bin"]
        )
        assert "==== File: /tmp/a.txt ====" in result
        assert "content a" in result
        assert "==== File: /tmp/b.bin (binary, base64) ====" in result

    def test_list_code_interpreters(self):
        mock_cp_client = MagicMock()
        mock_cp_client.list_code_interpreters.return_value = {
            "codeInterpreterSummaries": [
                {
                    "name": "my_interpreter",
                    "codeInterpreterId": "ci-123",
                    "status": "READY",
                    "type": "CUSTOM",
                },
            ]
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.list_code_interpreters(thread_id="test-thread")

        tool_spec._get_control_plane_client.assert_called_once()
        mock_cp_client.list_code_interpreters.assert_called_once_with(
            interpreter_type=None, max_results=10
        )
        assert "Found 1 code interpreter(s)" in result
        assert "my_interpreter" in result
        assert "ci-123" in result

    def test_list_code_interpreters_empty(self):
        mock_cp_client = MagicMock()
        mock_cp_client.list_code_interpreters.return_value = {
            "codeInterpreterSummaries": []
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.list_code_interpreters(thread_id="test-thread")

        assert "No code interpreters found" in result

    def test_list_code_interpreters_with_type_filter(self):
        mock_cp_client = MagicMock()
        mock_cp_client.list_code_interpreters.return_value = {
            "codeInterpreterSummaries": []
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        tool_spec.list_code_interpreters(
            interpreter_type="CUSTOM", thread_id="test-thread"
        )

        mock_cp_client.list_code_interpreters.assert_called_once_with(
            interpreter_type="CUSTOM", max_results=10
        )

    def test_list_code_interpreters_exception(self):
        mock_cp_client = MagicMock()
        mock_cp_client.list_code_interpreters.side_effect = Exception("API error")

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.list_code_interpreters(thread_id="test-thread")

        assert "Error listing code interpreters: API error" in result

    def test_create_code_interpreter(self):
        mock_cp_client = MagicMock()
        mock_cp_client.create_code_interpreter.return_value = {
            "codeInterpreterId": "ci-456",
            "status": "CREATING",
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.create_code_interpreter(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
            thread_id="test-thread",
        )

        tool_spec._get_control_plane_client.assert_called_once()
        mock_cp_client.create_code_interpreter.assert_called_once_with(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
            network_configuration={"networkMode": "PUBLIC"},
        )
        assert "ci-456" in result
        assert "CREATING" in result

    def test_create_code_interpreter_with_description(self):
        mock_cp_client = MagicMock()
        mock_cp_client.create_code_interpreter.return_value = {
            "codeInterpreterId": "ci-789",
            "status": "CREATING",
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.create_code_interpreter(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
            network_mode="VPC",
            description="A test interpreter",
            thread_id="test-thread",
        )

        mock_cp_client.create_code_interpreter.assert_called_once_with(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
            network_configuration={"networkMode": "VPC"},
            description="A test interpreter",
        )
        assert "ci-789" in result

    def test_create_code_interpreter_exception(self):
        mock_cp_client = MagicMock()
        mock_cp_client.create_code_interpreter.side_effect = Exception(
            "Permission denied"
        )

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.create_code_interpreter(
            name="test_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
            thread_id="test-thread",
        )

        assert "Error creating code interpreter: Permission denied" in result

    def test_create_code_interpreter_vpc_with_subnets(self):
        mock_cp_client = MagicMock()
        mock_cp_client.create_code_interpreter.return_value = {
            "codeInterpreterId": "ci-vpc",
            "status": "CREATING",
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.create_code_interpreter(
            name="vpc_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
            network_mode="VPC",
            subnet_ids=["subnet-abc", "subnet-def"],
            security_group_ids=["sg-123"],
            thread_id="test-thread",
        )

        mock_cp_client.create_code_interpreter.assert_called_once_with(
            name="vpc_interpreter",
            execution_role_arn="arn:aws:iam::123456789012:role/TestRole",
            network_configuration={
                "networkMode": "VPC",
                "vpcConfig": {
                    "subnets": ["subnet-abc", "subnet-def"],
                    "securityGroups": ["sg-123"],
                },
            },
        )
        assert "ci-vpc" in result

    def test_delete_code_interpreter(self):
        mock_cp_client = MagicMock()
        mock_cp_client.delete_code_interpreter.return_value = {
            "codeInterpreterId": "ci-123",
            "status": "DELETING",
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.delete_code_interpreter(
            interpreter_id="ci-123", thread_id="test-thread"
        )

        tool_spec._get_control_plane_client.assert_called_once()
        mock_cp_client.delete_code_interpreter.assert_called_once_with(
            interpreter_id="ci-123"
        )
        assert "ci-123" in result
        assert "DELETING" in result

    def test_delete_code_interpreter_exception(self):
        mock_cp_client = MagicMock()
        mock_cp_client.delete_code_interpreter.side_effect = Exception("Not found")

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.delete_code_interpreter(
            interpreter_id="ci-999", thread_id="test-thread"
        )

        assert "Error deleting code interpreter: Not found" in result

    def test_get_code_interpreter(self):
        mock_cp_client = MagicMock()
        mock_cp_client.get_code_interpreter.return_value = {
            "codeInterpreterId": "ci-123",
            "name": "my_interpreter",
            "status": "READY",
            "description": "A custom interpreter",
            "networkConfiguration": {"networkMode": "PUBLIC"},
        }

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.get_code_interpreter(
            interpreter_id="ci-123", thread_id="test-thread"
        )

        tool_spec._get_control_plane_client.assert_called_once()
        mock_cp_client.get_code_interpreter.assert_called_once_with(
            interpreter_id="ci-123"
        )
        assert "my_interpreter" in result
        assert "READY" in result
        assert "A custom interpreter" in result
        assert "PUBLIC" in result

    def test_get_code_interpreter_exception(self):
        mock_cp_client = MagicMock()
        mock_cp_client.get_code_interpreter.side_effect = Exception("Not found")

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_control_plane_client = MagicMock(return_value=mock_cp_client)

        result = tool_spec.get_code_interpreter(
            interpreter_id="ci-999", thread_id="test-thread"
        )

        assert "Error getting code interpreter: Not found" in result

    def test_clear_context(self):
        mock_code_interpreter = MagicMock()

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.clear_context(thread_id="test-thread")

        tool_spec._get_or_create_interpreter.assert_called_once_with(
            thread_id="test-thread"
        )
        mock_code_interpreter.clear_context.assert_called_once()
        assert "cleared successfully" in result

    def test_clear_context_exception(self):
        mock_code_interpreter = MagicMock()
        mock_code_interpreter.clear_context.side_effect = Exception("Session error")

        tool_spec = AgentCoreCodeInterpreterToolSpec()
        tool_spec._get_or_create_interpreter = MagicMock(
            return_value=mock_code_interpreter
        )

        result = tool_spec.clear_context(thread_id="test-thread")

        assert "Error clearing context: Session error" in result
