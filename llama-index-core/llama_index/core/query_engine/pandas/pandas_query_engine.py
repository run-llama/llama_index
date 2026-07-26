"""
Default query for PandasIndex.

DEPRECATED AND REMOVED: This class has been removed due to security concerns.

WARNING: The PandasQueryEngine previously provided LLM access to eval() and exec()
functions, enabling arbitrary code execution. Despite sandboxing attempts, Python's
introspection capabilities (MRO traversal, __builtins__ access, etc.) made it
impossible to provide a secure execution environment without external sandboxing
(containers, VMs, etc.).

For safe pandas operations, use explicit pandas query API methods instead of
dynamic code execution. If you need LLM-driven pandas operations, implement them
using the pandas query API with proper input validation, or run untrusted code
in an isolated sandbox environment (Docker, gVisor, etc.).

"""

from typing import Any


class PandasQueryEngine:
    """
    Pandas query engine.

    DEPRECATED AND REMOVED: This class has been removed due to security concerns.
    See module docstring for details and alternatives.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise DeprecationWarning(
            "PandasQueryEngine has been removed due to inherent security vulnerabilities.\n"
            "The sandboxed eval/exec environment could not be made secure against Python\n"
            "introspection attacks (MRO traversal, __builtins__ access, etc.).\n\n"
            "Alternatives:\n"
            "1. Use explicit pandas query API methods with proper input validation\n"
            "2. Run LLM-generated code in external sandboxed environments (Docker, gVisor, etc.)\n"
            "3. Use structured pandas operations without dynamic code execution\n\n"
            "Do not attempt to recreate this functionality without proper external sandboxing."
        )


# legacy
NLPandasQueryEngine = PandasQueryEngine
GPTNLPandasQueryEngine = PandasQueryEngine
