import ast
import copy
import signal
import sys
from contextlib import contextmanager
from types import CodeType, ModuleType
from typing import Any, Dict, Iterator, Mapping, Sequence, Union

ALLOWED_IMPORTS = {
    "math",
    "time",
    "datetime",
    "pandas",
    "polars",
    "scipy",
    "numpy",
    "matplotlib",
    "plotly",
    "seaborn",
}

# I/O methods on allowed libraries (pandas, numpy, polars) that can read/write
# files, databases, or the network.  These bypass the sandbox because the
# library objects are passed into locals, so we block them at the AST level.
_DANGEROUS_ATTR_CALLS = frozenset(
    {
        # pandas read / write
        "read_csv",
        "read_excel",
        "read_json",
        "read_html",
        "read_xml",
        "read_parquet",
        "read_feather",
        "read_orc",
        "read_stata",
        "read_sas",
        "read_spss",
        "read_sql",
        "read_sql_table",
        "read_sql_query",
        "read_gbq",
        "read_hdf",
        "read_pickle",
        "read_table",
        "read_fwf",
        "read_clipboard",
        "to_csv",
        "to_excel",
        "to_json",
        "to_html",
        "to_xml",
        "to_parquet",
        "to_feather",
        "to_orc",
        "to_stata",
        "to_hdf",
        "to_pickle",
        "to_sql",
        "to_gbq",
        "to_clipboard",
        "to_latex",
        "to_markdown",
        # polars read / write
        "scan_csv",
        "scan_parquet",
        "scan_ipc",
        "write_csv",
        "write_parquet",
        "write_ipc",
        # numpy file I/O
        "load",
        "save",
        "savez",
        "savez_compressed",
        "loadtxt",
        "savetxt",
        "genfromtxt",
        "fromfile",
        "tofile",
        # general dangerous calls
        "system",
        "popen",
    }
)

_DEFAULT_TIMEOUT_SECONDS = 30


@contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    """
    Raise TimeoutError if the wrapped block exceeds *seconds*.

    Uses SIGALRM on Unix.  On Windows (where SIGALRM is unavailable) the
    timeout is a no-op -- callers should layer additional protection there.
    """
    if sys.platform == "win32":
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Code execution exceeded {seconds}s time limit")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _restricted_import(
    name: str,
    globals: Union[Mapping[str, object], None] = None,
    locals: Union[Mapping[str, object], None] = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> ModuleType:
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
    restricted_globals = copy.deepcopy(ALLOWED_BUILTINS)
    if __globals:
        restricted_globals.update(__globals)
    return restricted_globals


class DunderVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_access_to_private_entity = False
        self.has_access_to_disallowed_builtin = False
        self.has_dangerous_io_call = False

        builtins = globals()["__builtins__"].keys()
        self._builtins = builtins

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            self.has_access_to_private_entity = True
        if node.id not in ALLOWED_BUILTINS and node.id in self._builtins:
            self.has_access_to_disallowed_builtin = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.has_access_to_private_entity = True
        if node.attr not in ALLOWED_BUILTINS and node.attr in self._builtins:
            self.has_access_to_disallowed_builtin = True
        if node.attr in _DANGEROUS_ATTR_CALLS:
            self.has_dangerous_io_call = True
        self.generic_visit(node)


def _contains_protected_access(code: str) -> bool:
    # do not allow imports
    imports_modules = False
    tree = ast.parse(code)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            imports_modules = True
        elif isinstance(node, ast.ImportFrom):
            imports_modules = True
        else:
            continue

    dunder_visitor = DunderVisitor()
    dunder_visitor.visit(tree)

    return (
        dunder_visitor.has_access_to_private_entity
        or dunder_visitor.has_access_to_disallowed_builtin
        or dunder_visitor.has_dangerous_io_call
        or imports_modules
    )


def _verify_source_safety(__source: Union[str, bytes, CodeType]) -> None:
    """
    Verify that the source is safe to execute. For now, this means that it
    does not contain any references to private or dunder methods.
    """
    if isinstance(__source, CodeType):
        raise RuntimeError("Direct execution of CodeType is forbidden!")
    if isinstance(__source, bytes):
        __source = __source.decode()
    if _contains_protected_access(__source):
        raise RuntimeError(
            "Execution of code containing references to private or dunder methods, "
            "disallowed builtins, dangerous I/O operations, or any imports, "
            "is forbidden!"
        )


def safe_eval(
    __source: Union[str, bytes, CodeType],
    __globals: Union[Dict[str, Any], None] = None,
    __locals: Union[Mapping[str, object], None] = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """Eval within a safe, time-limited global context."""
    _verify_source_safety(__source)
    with _time_limit(timeout_seconds):
        return eval(__source, _get_restricted_globals(__globals), __locals)


def safe_exec(
    __source: Union[str, bytes, CodeType],
    __globals: Union[Dict[str, Any], None] = None,
    __locals: Union[Mapping[str, object], None] = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Exec within a safe, time-limited global context."""
    _verify_source_safety(__source)
    with _time_limit(timeout_seconds):
        return exec(__source, _get_restricted_globals(__globals), __locals)
