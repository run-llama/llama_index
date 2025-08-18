import json
import os
import logging
from typing import Dict, Optional, List

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

DEFAULT_CODE_INTERPRETER_IDENTIFIER = "aws.codeinterpreter.v1"
DEFAULT_CODE_INTERPRETER_TIMEOUT = 900

logger = logging.getLogger(__name__)


def get_aws_region() -> str:
    """Get the AWS region from environment variables or use default."""
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"


def extract_output_from_stream(response):
    """
    Extract output from code interpreter response stream

    Args:
        response: Response from code interpreter execution

    Returns:
        Extracted output as string

    """
    output = []
    for event in response["stream"]:
        if "result" in event:
            result = event["result"]
            if "content" in result:
                for content_item in result["content"]:
                    if content_item["type"] == "text":
                        output.append(content_item["text"])
                    if content_item["type"] == "resource":
                        resource = content_item["resource"]
                        if "text" in resource:
                            file_path = resource["uri"].replace("file://", "")
                            file_content = resource["text"]
                            output.append(
                                f"==== File: {file_path} ====\n{file_content}\n"
                            )
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
    ]

    def __init__(self, region: Optional[str] = None) -> None:
        """
        Initialize the AWS Bedrock AgentCore Code Interpreter Tool Spec.

        Args:
            region (Optional[str]): AWS region to use for Bedrock AgentCore services.
                If not provided, will try to get it from environment variables.

        """
        self.region = region if region is not None else get_aws_region()
        self._code_interpreters: Dict[str, CodeInterpreter] = {}

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
        code_interpreter = CodeInterpreter(region=self.region)
        code_interpreter.start()
        logger.info(
            f"Started code interpreter with session_id:{code_interpreter.session_id} for thread:{thread_id}"
        )

        # Store the interpreter
        self._code_interpreters[thread_id] = code_interpreter
        return code_interpreter

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
        # Use the synchronous version as the underlying API is thread-safe
        return self.execute_code(
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
        # Use the synchronous version as the underlying API is thread-safe
        return self.execute_command(command=command, thread_id=thread_id)

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
        # Use the synchronous version as the underlying API is thread-safe
        return self.read_files(paths=paths, thread_id=thread_id)

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
        # Use the synchronous version as the underlying API is thread-safe
        return self.list_files(directory_path=directory_path, thread_id=thread_id)

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
        # Use the synchronous version as the underlying API is thread-safe
        return self.delete_files(paths=paths, thread_id=thread_id)

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
        # Use the synchronous version as the underlying API is thread-safe
        return self.write_files(files=files, thread_id=thread_id)

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
        # Use the synchronous version as the underlying API is thread-safe
        return self.start_command(command=command, thread_id=thread_id)

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
        # Use the synchronous version as the underlying API is thread-safe
        return self.get_task(task_id=task_id, thread_id=thread_id)

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
        # Use the synchronous version as the underlying API is thread-safe
        return self.stop_task(task_id=task_id, thread_id=thread_id)

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
