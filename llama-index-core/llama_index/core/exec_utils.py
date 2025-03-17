"""Utilities for safe code execution and evaluation.

This module provides utilities for safely executing and evaluating code by restricting
access to potentially dangerous operations. It includes a set of allowed imports and
builtins, and prevents access to private methods and attributes.
"""
import ast
import copy
from types import CodeType, ModuleType
from typing import Any, Dict, Mapping, Sequence, Union

ALLOWED_IMPORTS = {
    "math",
    "time",
    "datetime",
    "pandas",
    "scipy",
    "numpy",
    "matplotlib",
    "plotly",
    "seaborn",
}


def _restricted_import(
    name: str,
    globals: Union[Mapping[str, object], None] = None,
    locals: Union[Mapping[str, object], None] = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> ModuleType:
    """Restrict imports to a set of allowed modules.

    Args:
        name (str): The name of the module to import.
        globals (Union[Mapping[str, object], None], optional): Global namespace. Defaults to None.
        locals (Union[Mapping[str, object], None], optional): Local namespace. Defaults to None.
        fromlist (Sequence[str], optional): List of names to import. Defaults to ().
        level (int, optional): The level of relative import. Defaults to 0.

    Returns:
        ModuleType: The imported module if allowed.

    Raises:
        ImportError: If the module is not in the allowed list.
    """
    if name in ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is not allowed")


ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    # Constants
    "True": True,
    "False": False,
    "None": None,
    "__import__": _restricted_import,
}


def _get_restricted_globals(__globals: Union[dict, None]) -> Any:
    """Create a restricted globals dictionary with only allowed builtins.

    Args:
        __globals (Union[dict, None]): Additional globals to include.

    Returns:
        Any: A dictionary containing only allowed builtins and provided globals.
    """
    restricted_globals = copy.deepcopy(ALLOWED_BUILTINS)
    if __globals:
        restricted_globals.update(__globals)
    return restricted_globals


vulnerable_code_snippets = [
    "os.",
]


class DunderVisitor(ast.NodeVisitor):
    """AST visitor that checks for access to private entities and disallowed builtins."""

    def __init__(self) -> None:
        self.has_access_to_private_entity = False
        self.has_access_to_disallowed_builtin = False

        builtins = globals()["__builtins__"].keys()
        self._builtins = builtins

    def visit_Name(self, node: ast.Name) -> None:
        """Visit a name node in the AST.

        Args:
            node (ast.Name): The AST name node to visit.
        """
        if node.id.startswith("_"):
            self.has_access_to_private_entity = True
        if node.id not in ALLOWED_BUILTINS and node.id in self._builtins:
            self.has_access_to_disallowed_builtin = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit an attribute node in the AST.

        Args:
            node (ast.Attribute): The AST attribute node to visit.
        """
        if node.attr.startswith("_"):
            self.has_access_to_private_entity = True
        if node.attr not in ALLOWED_BUILTINS and node.attr in self._builtins:
            self.has_access_to_disallowed_builtin = True
        self.generic_visit(node)


def _contains_protected_access(code: str) -> bool:
    """Check if code contains protected access or disallowed operations.

    Args:
        code (str): The code string to check.

    Returns:
        bool: True if code contains protected access or disallowed operations.
    """
    # do not allow imports
    tree = ast.parse(code)
    imports_modules = any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.iter_child_nodes(tree)
    )

    dunder_visitor = DunderVisitor()
    dunder_visitor.visit(tree)

    for vulnerable_code_snippet in vulnerable_code_snippets:
        if vulnerable_code_snippet in code:
            dunder_visitor.has_access_to_disallowed_builtin = True

    return (
        dunder_visitor.has_access_to_private_entity
        or dunder_visitor.has_access_to_disallowed_builtin
        or imports_modules
    )


def _verify_source_safety(__source: Union[str, bytes, CodeType]) -> None:
    """Verify that the source code is safe to execute.

    Args:
        __source (Union[str, bytes, CodeType]): The source code to verify.

    Raises:
        RuntimeError: If the code contains unsafe operations.
    """
    if isinstance(__source, CodeType):
        raise RuntimeError("Direct execution of CodeType is forbidden!")
    if isinstance(__source, bytes):
        __source = __source.decode()
    if _contains_protected_access(__source):
        raise RuntimeError(
            "Execution of code containing references to private or dunder methods, "
            "disallowed builtins, or any imports, is forbidden!"
        )


def safe_eval(
    __source: Union[str, bytes, CodeType],
    __globals: Union[Dict[str, Any], None] = None,
    __locals: Union[Mapping[str, object], None] = None,
) -> Any:
    """Safely evaluate an expression within a restricted context.

    Args:
        __source (Union[str, bytes, CodeType]): The source code to evaluate.
        __globals (Union[Dict[str, Any], None], optional): Global namespace. Defaults to None.
        __locals (Union[Mapping[str, object], None], optional): Local namespace. Defaults to None.

    Returns:
        Any: The result of evaluating the expression.
    """
    _verify_source_safety(__source)
    return eval(__source, _get_restricted_globals(__globals), __locals)


def safe_exec(
    __source: Union[str, bytes, CodeType],
    __globals: Union[Dict[str, Any], None] = None,
    __locals: Union[Mapping[str, object], None] = None,
) -> None:
    """Safely execute code within a restricted context.

    Args:
        __source (Union[str, bytes, CodeType]): The source code to execute.
        __globals (Union[Dict[str, Any], None], optional): Global namespace.
            Defaults to None.
        __locals (Union[Mapping[str, object], None], optional): Local namespace.
            Defaults to None.
    """
    _verify_source_safety(__source)
    return exec(__source, _get_restricted_globals(__globals), __locals)
