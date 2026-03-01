import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

from llama_index.tools.aws_bedrock_agentcore.utils import get_aws_region

DEFAULT_CODE_INTERPRETER_IDENTIFIER = "aws.codeinterpreter.v1"
DEFAULT_CODE_INTERPRETER_TIMEOUT = 900

logger = logging.getLogger(__name__)


def extract_output_from_stream(response):
    """
    Extract output from code interpreter response stream

    Args:
        response: Response from code interpreter execution

    Returns:
        Extracted output as string

    """
    output = []
    for event in response.get("stream", []):
        result = event.get("result", {})
        for content_item in result.get("content", []):
            if content_item.get("type") == "text":
                output.append(content_item.get("text", ""))
            if content_item.get("type") == "resource":
                resource = content_item.get("resource", {})
                if "text" in resource:
                    file_path = resource.get("uri", "").replace("file://", "")
                    file_content = resource["text"]
                    output.append(f"==== File: {file_path} ====\n{file_content}\n")
                else:
                    output.append(json.dumps(resource))

    return "\n".join(output)


class AgentCoreCodeInterpreterToolSpec(BaseToolSpec):
    """
    AWS Bedrock AgentCore Code Interpreter Tool Spec.

    This toolkit provides a set of tools for working with a remote code interpreter environment:

    * execute_code - Run code in various languages (primarily Python)
    * execute_command - Run shell commands
    * read_files - Read content of files in the environment
    * list_files - List files in directories
    * delete_files - Remove files from the environment
    * write_files - Create or update files
    * start_command - Start long-running commands asynchronously
    * get_task - Check status of async tasks
    * stop_task - Stop running tasks

    The toolkit lazily initializes the code interpreter session on first use.
    It supports multiple threads by maintaining separate code interpreter sessions for each thread ID.
    """

    spec_functions = [
        ("execute_code", "aexecute_code"),
        ("execute_command", "aexecute_command"),
        ("read_files", "aread_files"),
        ("list_files", "alist_files"),
        ("delete_files", "adelete_files"),
        ("write_files", "awrite_files"),
        ("start_command", "astart_command"),
        ("get_task", "aget_task"),
        ("stop_task", "astop_task"),
        ("upload_file", "aupload_file"),
        ("upload_files", "aupload_files"),
        ("install_packages", "ainstall_packages"),
        ("download_file", "adownload_file"),
        ("download_files", "adownload_files"),
        ("list_code_interpreters", "alist_code_interpreters"),
        ("create_code_interpreter", "acreate_code_interpreter"),
        ("delete_code_interpreter", "adelete_code_interpreter"),
        ("get_code_interpreter", "aget_code_interpreter"),
        ("clear_context", "aclear_context"),
    ]

    def __init__(
        self,
        region: Optional[str] = None,
        identifier: Optional[str] = None,
    ) -> None:
        """
        Initialize the AWS Bedrock AgentCore Code Interpreter Tool Spec.

        Args:
            region (Optional[str]): AWS region to use for Bedrock AgentCore services.
                If not provided, will try to get it from environment variables.
            identifier (Optional[str]): Custom code interpreter identifier for
                VPC-enabled resources. If not provided, uses the default identifier.

        """
        self.region = region if region is not None else get_aws_region()
        self._identifier = identifier
        self._code_interpreters: Dict[str, CodeInterpreter] = {}
        self._cp_ci_client: Optional[CodeInterpreter] = None

    def _get_or_create_interpreter(self, thread_id: str = "default") -> CodeInterpreter:
        """
        Get or create a code interpreter for the specified thread.

        Args:
            thread_id: Thread ID for the code interpreter session

        Returns:
            CodeInterpreter instance

        """
        if thread_id in self._code_interpreters:
            return self._code_interpreters[thread_id]

        # Create a new code interpreter for this thread
        code_interpreter = CodeInterpreter(
            region=self.region, integration_source="llamaindex"
        )
        start_kwargs = {}
        if self._identifier is not None:
            start_kwargs["identifier"] = self._identifier
        code_interpreter.start(**start_kwargs)
        logger.info(
            f"Started code interpreter with session_id:{code_interpreter.session_id} for thread:{thread_id}"
        )

        # Store the interpreter
        self._code_interpreters[thread_id] = code_interpreter
        return code_interpreter

    def _get_control_plane_client(self) -> CodeInterpreter:
        """
        Get or create a code interpreter client for control-plane operations only.

        This client is used for account-level operations (list, create, delete, get)
        that do not require starting a session.
        """
        if self._cp_ci_client is None:
            self._cp_ci_client = CodeInterpreter(
                region=self.region, integration_source="llamaindex"
            )
        return self._cp_ci_client

    def execute_code(
        self,
        code: str,
        language: str = "python",
        clear_context: bool = False,
        thread_id: str = "default",
    ) -> str:
        """
        Execute code in the code interpreter sandbox (synchronous version).

        Args:
            code (str): The code to execute.
            language (str): The programming language of the code. Default is "python".
            clear_context (bool): Whether to clear execution context. Default is False.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the code execution.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # Execute code
            response = code_interpreter.invoke(
                method="executeCode",
                params={
                    "code": code,
                    "language": language,
                    "clearContext": clear_context,
                },
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error executing code: {e!s}"

    async def aexecute_code(
        self,
        code: str,
        language: str = "python",
        clear_context: bool = False,
        thread_id: str = "default",
    ) -> str:
        """
        Execute code in the code interpreter sandbox (asynchronous version).

        Args:
            code (str): The code to execute.
            language (str): The programming language of the code. Default is "python".
            clear_context (bool): Whether to clear execution context. Default is False.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the code execution.

        """
        return await asyncio.to_thread(
            self.execute_code,
            code=code,
            language=language,
            clear_context=clear_context,
            thread_id=thread_id,
        )

    def execute_command(
        self,
        command: str,
        thread_id: str = "default",
    ) -> str:
        """
        Execute a shell command in the code interpreter sandbox (synchronous version).

        Args:
            command (str): The command to execute.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the command execution.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # Execute command
            response = code_interpreter.invoke(
                method="executeCommand", params={"command": command}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error executing command: {e!s}"

    async def aexecute_command(
        self,
        command: str,
        thread_id: str = "default",
    ) -> str:
        """
        Execute a shell command in the code interpreter sandbox (asynchronous version).

        Args:
            command (str): The command to execute.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the command execution.

        """
        return await asyncio.to_thread(
            self.execute_command, command=command, thread_id=thread_id
        )

    def read_files(
        self,
        paths: List[str],
        thread_id: str = "default",
    ) -> str:
        """
        Read content of files in the environment (synchronous version).

        Args:
            paths (List[str]): List of file paths to read.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The content of the files.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # Read files
            response = code_interpreter.invoke(
                method="readFiles", params={"paths": paths}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error reading files: {e!s}"

    async def aread_files(
        self,
        paths: List[str],
        thread_id: str = "default",
    ) -> str:
        """
        Read content of files in the environment (asynchronous version).

        Args:
            paths (List[str]): List of file paths to read.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The content of the files.

        """
        return await asyncio.to_thread(
            self.read_files, paths=paths, thread_id=thread_id
        )

    def list_files(
        self,
        directory_path: str = "",
        thread_id: str = "default",
    ) -> str:
        """
        List files in directories in the environment (synchronous version).

        Args:
            directory_path (str): Path to the directory to list. Default is current directory.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The list of files.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # List files
            response = code_interpreter.invoke(
                method="listFiles", params={"directoryPath": directory_path}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error listing files: {e!s}"

    async def alist_files(
        self,
        directory_path: str = "",
        thread_id: str = "default",
    ) -> str:
        """
        List files in directories in the environment (asynchronous version).

        Args:
            directory_path (str): Path to the directory to list. Default is current directory.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The list of files.

        """
        return await asyncio.to_thread(
            self.list_files, directory_path=directory_path, thread_id=thread_id
        )

    def delete_files(
        self,
        paths: List[str],
        thread_id: str = "default",
    ) -> str:
        """
        Remove files from the environment (synchronous version).

        Args:
            paths (List[str]): List of file paths to delete.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the delete operation.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # Remove files
            response = code_interpreter.invoke(
                method="removeFiles", params={"paths": paths}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error deleting files: {e!s}"

    async def adelete_files(
        self,
        paths: List[str],
        thread_id: str = "default",
    ) -> str:
        """
        Remove files from the environment (asynchronous version).

        Args:
            paths (List[str]): List of file paths to delete.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the delete operation.

        """
        return await asyncio.to_thread(
            self.delete_files, paths=paths, thread_id=thread_id
        )

    def write_files(
        self,
        files: List[Dict[str, str]],
        thread_id: str = "default",
    ) -> str:
        """
        Create or update files in the environment (synchronous version).

        Args:
            files (List[Dict[str, str]]): List of dictionaries with path and text fields.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the write operation.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # Write files
            response = code_interpreter.invoke(
                method="writeFiles", params={"content": files}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error writing files: {e!s}"

    async def awrite_files(
        self,
        files: List[Dict[str, str]],
        thread_id: str = "default",
    ) -> str:
        """
        Create or update files in the environment (asynchronous version).

        Args:
            files (List[Dict[str, str]]): List of dictionaries with path and text fields.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the write operation.

        """
        return await asyncio.to_thread(
            self.write_files, files=files, thread_id=thread_id
        )

    def start_command(
        self,
        command: str,
        thread_id: str = "default",
    ) -> str:
        """
        Start a long-running command asynchronously (synchronous version).

        Args:
            command (str): The command to execute asynchronously.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The task ID and status.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # Start command execution
            response = code_interpreter.invoke(
                method="startCommandExecution", params={"command": command}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error starting command: {e!s}"

    async def astart_command(
        self,
        command: str,
        thread_id: str = "default",
    ) -> str:
        """
        Start a long-running command asynchronously (asynchronous version).

        Args:
            command (str): The command to execute asynchronously.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The task ID and status.

        """
        return await asyncio.to_thread(
            self.start_command, command=command, thread_id=thread_id
        )

    def get_task(
        self,
        task_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Check status of an async task (synchronous version).

        Args:
            task_id (str): The ID of the task to check.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The task status.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # Get task status
            response = code_interpreter.invoke(
                method="getTask", params={"taskId": task_id}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error getting task status: {e!s}"

    async def aget_task(
        self,
        task_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Check status of an async task (asynchronous version).

        Args:
            task_id (str): The ID of the task to check.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The task status.

        """
        return await asyncio.to_thread(
            self.get_task, task_id=task_id, thread_id=thread_id
        )

    def stop_task(
        self,
        task_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Stop a running task (synchronous version).

        Args:
            task_id (str): The ID of the task to stop.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the stop operation.

        """
        try:
            # Get or create code interpreter
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)

            # Stop task
            response = code_interpreter.invoke(
                method="stopTask", params={"taskId": task_id}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error stopping task: {e!s}"

    async def astop_task(
        self,
        task_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Stop a running task (asynchronous version).

        Args:
            task_id (str): The ID of the task to stop.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The result of the stop operation.

        """
        return await asyncio.to_thread(
            self.stop_task, task_id=task_id, thread_id=thread_id
        )

    def upload_file(
        self,
        path: str,
        content: str,
        description: str = "",
        thread_id: str = "default",
    ) -> str:
        """
        Upload a file to the code interpreter sandbox (synchronous version).

        Args:
            path (str): Relative file path in the sandbox.
            content (str): File content as a string.
            description (str): Semantic description of the file. Default is "".
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: Confirmation message with the uploaded file path.

        """
        try:
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)
            code_interpreter.upload_file(
                path=path, content=content, description=description
            )
            return f"Uploaded file to {path}"
        except Exception as e:
            return f"Error uploading file: {e!s}"

    async def aupload_file(
        self,
        path: str,
        content: str,
        description: str = "",
        thread_id: str = "default",
    ) -> str:
        """
        Upload a file to the code interpreter sandbox (asynchronous version).

        Args:
            path (str): Relative file path in the sandbox.
            content (str): File content as a string.
            description (str): Semantic description of the file. Default is "".
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: Confirmation message with the uploaded file path.

        """
        return await asyncio.to_thread(
            self.upload_file,
            path=path,
            content=content,
            description=description,
            thread_id=thread_id,
        )

    def upload_files(
        self,
        files: List[Dict[str, str]],
        thread_id: str = "default",
    ) -> str:
        """
        Upload multiple files to the code interpreter sandbox (synchronous version).

        Args:
            files (List[Dict[str, str]]): List of file specifications, each with
                'path', 'content', and optional 'description' keys.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: Confirmation message with the number of files uploaded.

        """
        try:
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)
            code_interpreter.upload_files(files=files)
            return f"Uploaded {len(files)} file(s)"
        except Exception as e:
            return f"Error uploading files: {e!s}"

    async def aupload_files(
        self,
        files: List[Dict[str, str]],
        thread_id: str = "default",
    ) -> str:
        """
        Upload multiple files to the code interpreter sandbox (asynchronous version).

        Args:
            files (List[Dict[str, str]]): List of file specifications, each with
                'path', 'content', and optional 'description' keys.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: Confirmation message with the number of files uploaded.

        """
        return await asyncio.to_thread(
            self.upload_files, files=files, thread_id=thread_id
        )

    def install_packages(
        self,
        packages: List[str],
        upgrade: bool = False,
        thread_id: str = "default",
    ) -> str:
        """
        Install Python packages in the code interpreter sandbox (synchronous version).

        Args:
            packages (List[str]): List of package names to install. Can include version
                specifiers (e.g., 'pandas>=2.0').
            upgrade (bool): Whether to upgrade existing packages. Default is False.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The pip install output (stdout/stderr).

        """
        try:
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)
            result = code_interpreter.install_packages(
                packages=packages, upgrade=upgrade
            )
            return str(result)
        except Exception as e:
            return f"Error installing packages: {e!s}"

    async def ainstall_packages(
        self,
        packages: List[str],
        upgrade: bool = False,
        thread_id: str = "default",
    ) -> str:
        """
        Install Python packages in the code interpreter sandbox (asynchronous version).

        Args:
            packages (List[str]): List of package names to install. Can include version
                specifiers (e.g., 'pandas>=2.0').
            upgrade (bool): Whether to upgrade existing packages. Default is False.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The pip install output (stdout/stderr).

        """
        return await asyncio.to_thread(
            self.install_packages,
            packages=packages,
            upgrade=upgrade,
            thread_id=thread_id,
        )

    def download_file(
        self,
        path: str,
        thread_id: str = "default",
    ) -> str:
        """
        Download a file from the code interpreter sandbox (synchronous version).

        Args:
            path (str): Path to the file in the sandbox.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The file content as text, or base64-encoded string for binary files.

        """
        try:
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)
            content = code_interpreter.download_file(path=path)
            if isinstance(content, bytes):
                encoded = base64.b64encode(content).decode("utf-8")
                return f"[base64 encoded binary file: {path}]\n{encoded}"
            return content
        except Exception as e:
            return f"Error downloading file: {e!s}"

    async def adownload_file(
        self,
        path: str,
        thread_id: str = "default",
    ) -> str:
        """
        Download a file from the code interpreter sandbox (asynchronous version).

        Args:
            path (str): Path to the file in the sandbox.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: The file content as text, or base64-encoded string for binary files.

        """
        return await asyncio.to_thread(
            self.download_file, path=path, thread_id=thread_id
        )

    def download_files(
        self,
        paths: List[str],
        thread_id: str = "default",
    ) -> str:
        """
        Download multiple files from the code interpreter sandbox (synchronous version).

        Args:
            paths (List[str]): List of file paths in the sandbox.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: Formatted output with each file's content.

        """
        try:
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)
            results = code_interpreter.download_files(paths=paths)
            output = []
            for file_path, content in results.items():
                if isinstance(content, bytes):
                    encoded = base64.b64encode(content).decode("utf-8")
                    output.append(
                        f"==== File: {file_path} (binary, base64) ====\n{encoded}"
                    )
                else:
                    output.append(f"==== File: {file_path} ====\n{content}")
            return "\n\n".join(output)
        except Exception as e:
            return f"Error downloading files: {e!s}"

    async def adownload_files(
        self,
        paths: List[str],
        thread_id: str = "default",
    ) -> str:
        """
        Download multiple files from the code interpreter sandbox (asynchronous version).

        Args:
            paths (List[str]): List of file paths in the sandbox.
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: Formatted output with each file's content.

        """
        return await asyncio.to_thread(
            self.download_files, paths=paths, thread_id=thread_id
        )

    def list_code_interpreters(
        self,
        interpreter_type: Optional[str] = None,
        max_results: int = 10,
        thread_id: str = "default",
    ) -> str:
        """
        List all code interpreters in the account (synchronous version).

        Args:
            interpreter_type (Optional[str]): Filter by type: "SYSTEM" or "CUSTOM".
            max_results (int): Maximum results to return (1-100). Default is 10.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Formatted list of code interpreter summaries.

        """
        try:
            code_interpreter = self._get_control_plane_client()
            response = code_interpreter.list_code_interpreters(
                interpreter_type=interpreter_type, max_results=max_results
            )
            summaries = response.get("codeInterpreterSummaries", [])
            if not summaries:
                return "No code interpreters found."
            lines = []
            for ci in summaries:
                lines.append(
                    f"- {ci.get('name', 'N/A')} (ID: {ci.get('codeInterpreterId', 'N/A')}, "
                    f"Status: {ci.get('status', 'N/A')}, Type: {ci.get('type', 'N/A')})"
                )
            return f"Found {len(summaries)} code interpreter(s):\n" + "\n".join(lines)
        except Exception as e:
            return f"Error listing code interpreters: {e!s}"

    async def alist_code_interpreters(
        self,
        interpreter_type: Optional[str] = None,
        max_results: int = 10,
        thread_id: str = "default",
    ) -> str:
        """
        List all code interpreters in the account (asynchronous version).

        Args:
            interpreter_type (Optional[str]): Filter by type: "SYSTEM" or "CUSTOM".
            max_results (int): Maximum results to return (1-100). Default is 10.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Formatted list of code interpreter summaries.

        """
        return await asyncio.to_thread(
            self.list_code_interpreters,
            interpreter_type=interpreter_type,
            max_results=max_results,
            thread_id=thread_id,
        )

    def create_code_interpreter(
        self,
        name: str,
        execution_role_arn: str,
        network_mode: str = "PUBLIC",
        description: str = "",
        subnet_ids: Optional[List[str]] = None,
        security_group_ids: Optional[List[str]] = None,
        thread_id: str = "default",
    ) -> str:
        """
        Create a custom code interpreter with specific configuration (synchronous version).

        Args:
            name (str): Name for the interpreter. Must match pattern [a-zA-Z][a-zA-Z0-9_]{0,47}.
            execution_role_arn (str): IAM role ARN with permissions for interpreter operations.
            network_mode (str): Network mode: "PUBLIC" or "VPC". Default is "PUBLIC".
            description (str): Description of the interpreter. Default is "".
            subnet_ids (Optional[List[str]]): Subnet IDs for VPC mode.
            security_group_ids (Optional[List[str]]): Security group IDs for VPC mode.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Confirmation with interpreter ID and status.

        """
        try:
            code_interpreter = self._get_control_plane_client()
            network_config: Dict[str, Any] = {"networkMode": network_mode}
            if subnet_ids or security_group_ids:
                vpc_config: Dict[str, Any] = {}
                if subnet_ids:
                    vpc_config["subnets"] = subnet_ids
                if security_group_ids:
                    vpc_config["securityGroups"] = security_group_ids
                network_config["vpcConfig"] = vpc_config
            kwargs: Dict[str, Any] = {
                "name": name,
                "execution_role_arn": execution_role_arn,
                "network_configuration": network_config,
            }
            if description:
                kwargs["description"] = description
            response = code_interpreter.create_code_interpreter(**kwargs)
            interpreter_id = response.get("codeInterpreterId", "unknown")
            status = response.get("status", "unknown")
            return f"Code interpreter created (ID: {interpreter_id}, Status: {status})"
        except Exception as e:
            return f"Error creating code interpreter: {e!s}"

    async def acreate_code_interpreter(
        self,
        name: str,
        execution_role_arn: str,
        network_mode: str = "PUBLIC",
        description: str = "",
        subnet_ids: Optional[List[str]] = None,
        security_group_ids: Optional[List[str]] = None,
        thread_id: str = "default",
    ) -> str:
        """
        Create a custom code interpreter with specific configuration (asynchronous version).

        Args:
            name (str): Name for the interpreter. Must match pattern [a-zA-Z][a-zA-Z0-9_]{0,47}.
            execution_role_arn (str): IAM role ARN with permissions for interpreter operations.
            network_mode (str): Network mode: "PUBLIC" or "VPC". Default is "PUBLIC".
            description (str): Description of the interpreter. Default is "".
            subnet_ids (Optional[List[str]]): Subnet IDs for VPC mode.
            security_group_ids (Optional[List[str]]): Security group IDs for VPC mode.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Confirmation with interpreter ID and status.

        """
        return await asyncio.to_thread(
            self.create_code_interpreter,
            name=name,
            execution_role_arn=execution_role_arn,
            network_mode=network_mode,
            description=description,
            subnet_ids=subnet_ids,
            security_group_ids=security_group_ids,
            thread_id=thread_id,
        )

    def delete_code_interpreter(
        self,
        interpreter_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Delete a custom code interpreter (synchronous version).

        Args:
            interpreter_id (str): The code interpreter identifier to delete.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Confirmation of deletion.

        """
        try:
            code_interpreter = self._get_control_plane_client()
            response = code_interpreter.delete_code_interpreter(
                interpreter_id=interpreter_id
            )
            status = response.get("status", "unknown")
            return f"Code interpreter '{interpreter_id}' deleted (Status: {status})"
        except Exception as e:
            return f"Error deleting code interpreter: {e!s}"

    async def adelete_code_interpreter(
        self,
        interpreter_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Delete a custom code interpreter (asynchronous version).

        Args:
            interpreter_id (str): The code interpreter identifier to delete.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Confirmation of deletion.

        """
        return await asyncio.to_thread(
            self.delete_code_interpreter,
            interpreter_id=interpreter_id,
            thread_id=thread_id,
        )

    def get_code_interpreter(
        self,
        interpreter_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Get detailed information about a code interpreter (synchronous version).

        Args:
            interpreter_id (str): The code interpreter identifier.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Interpreter details including name, status, and configuration.

        """
        try:
            code_interpreter = self._get_control_plane_client()
            response = code_interpreter.get_code_interpreter(
                interpreter_id=interpreter_id
            )
            name = response.get("name", "N/A")
            status = response.get("status", "N/A")
            desc = response.get("description", "")
            result = f"Code interpreter '{interpreter_id}':\n"
            result += f"  Name: {name}\n"
            result += f"  Status: {status}\n"
            if desc:
                result += f"  Description: {desc}\n"
            network = response.get("networkConfiguration", {})
            if network:
                result += f"  Network mode: {network.get('networkMode', 'N/A')}"
            return result
        except Exception as e:
            return f"Error getting code interpreter: {e!s}"

    async def aget_code_interpreter(
        self,
        interpreter_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Get detailed information about a code interpreter (asynchronous version).

        Args:
            interpreter_id (str): The code interpreter identifier.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Interpreter details including name, status, and configuration.

        """
        return await asyncio.to_thread(
            self.get_code_interpreter,
            interpreter_id=interpreter_id,
            thread_id=thread_id,
        )

    def clear_context(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Clear all variable state in the Python execution context (synchronous version).

        This resets the interpreter to a fresh state, removing all previously defined
        variables, imports, and function definitions.

        Args:
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: Confirmation that the context was cleared.

        """
        try:
            code_interpreter = self._get_or_create_interpreter(thread_id=thread_id)
            code_interpreter.clear_context()
            return "Python execution context cleared successfully."
        except Exception as e:
            return f"Error clearing context: {e!s}"

    async def aclear_context(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Clear all variable state in the Python execution context (asynchronous version).

        Args:
            thread_id (str): Thread ID for the code interpreter session. Default is "default".

        Returns:
            str: Confirmation that the context was cleared.

        """
        return await asyncio.to_thread(self.clear_context, thread_id=thread_id)

    async def cleanup(self, thread_id: Optional[str] = None) -> None:
        """
        Clean up resources

        Args:
            thread_id: Optional thread ID to clean up. If None, cleans up all sessions.

        """
        if thread_id:
            # Clean up a specific thread's session
            if thread_id in self._code_interpreters:
                try:
                    self._code_interpreters[thread_id].stop()
                    del self._code_interpreters[thread_id]
                    logger.info(
                        f"Code interpreter session for thread {thread_id} cleaned up"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error stopping code interpreter for thread {thread_id}: {e}"
                    )
        else:
            # Clean up all sessions
            thread_ids = list(self._code_interpreters.keys())
            for tid in thread_ids:
                try:
                    self._code_interpreters[tid].stop()
                except Exception as e:
                    logger.warning(
                        f"Error stopping code interpreter for thread {tid}: {e}"
                    )

            self._code_interpreters = {}
            logger.info("All code interpreter sessions cleaned up")
