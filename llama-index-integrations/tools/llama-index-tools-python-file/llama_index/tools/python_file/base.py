import ast
from typing import List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class PythonFileToolSpec(BaseToolSpec):
    spec_functions = ["function_definitions", "get_function", "get_functions"]

    def __init__(self, file_name: str) -> None:
        f = open(file_name).read()
        self.tree = ast.parse(f)

    def function_definitions(self, external: Optional[bool] = True) -> str:
        """
        Use this function to get the name and arguments of all function definitions in the python file.

        Args:
            external (Optional[bool]): Defaults to true. If false, this function will also return functions that start with _

        """
        functions = ""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if external and node.name.startswith("_"):
                    continue
                functions += f"""
name: {node.name}
arguments: {ast.dump(node.args)}
                    """
        return functions

    def get_function(self, name: str) -> str:
        """
        Use this function to get the name and arguments of a single function definition in the python file.

        Args:
            name (str): The name of the function to retrieve

        """
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == name:
                    return f"""
name: {node.name}
arguments: {ast.dump(node.args)}
docstring: {ast.get_docstring(node)}
                        """
        return None

    def get_functions(self, names: List[str]) -> str:
        """
        Use this function to get the name and arguments of a list of function definition in the python file.

        Args:
            name (List[str]): The names of the functions to retrieve

        """
        functions = ""
        for name in names:
            functions += self.get_function(name) + "\n"
        return functions
