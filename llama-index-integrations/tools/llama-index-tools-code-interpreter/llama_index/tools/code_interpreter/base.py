"""Code Interpreter tool spec."""

import subprocess
import sys

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class CodeInterpreterToolSpec(BaseToolSpec):
    """Code Interpreter tool spec.

    WARNING: This tool provides the Agent access to the `subprocess.run` command.
    Arbitrary code execution is possible on the machine running this tool.
    This tool is not recommended to be used in a production setting, and would require heavy sandboxing or virtual machines

    """

    spec_functions = ["code_interpreter"]

    def code_interpreter(self, code: str):
        """
        A function to execute python code, and return the stdout and stderr.

        You should import any libraries that you wish to use. You have access to any libraries the user has installed.

        The code passed to this function is executed in isolation. It should be complete at the time it is passed to this function.

        You should interpret the output and errors returned from this function, and attempt to fix any problems.
        If you cannot fix the error, show the code to the user and ask for help

        It is not possible to return graphics or other complicated data from this function. If the user cannot see the output, save it to a file and tell the user.
        """
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        return f"StdOut:\n{result.stdout}\nStdErr:\n{result.stderr}"
