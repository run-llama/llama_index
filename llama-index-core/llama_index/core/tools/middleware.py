"""Built-in tool middleware implementations."""

from typing import Any, Dict, Optional, Set

from llama_index.core.tools.types import ToolMiddleware

# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class ParameterInjectionMiddleware(ToolMiddleware):
    """
    Middleware that injects trusted parameters into tool inputs.

    Unlike ``partial_params``, this middleware can **enforce** parameter values
    so that the LLM cannot override them. Parameters listed in ``enforce``
    will always be set to the values provided in ``params``, regardless of
    what the LLM supplies. Parameters not in ``enforce`` act as defaults
    that the LLM *can* override.

    Args:
        params: Dict of parameter names to values to inject.
        enforce: Optional set of parameter names whose values must not be
            overridden by the LLM. If ``None``, **all** params are enforced.

    Example:
        >>> middleware = ParameterInjectionMiddleware(
        ...     params={"api_key": "trusted-key", "user_id": "user-123"},
        ...     enforce={"api_key"},  # api_key cannot be overridden
        ... )
        >>> tool = FunctionTool.from_defaults(my_fn, middleware=[middleware])

    """

    def __init__(
        self,
        params: Dict[str, Any],
        enforce: Optional[Set[str]] = None,
    ) -> None:
        self._params = params
        # If enforce is None, enforce all params
        self._enforce = enforce if enforce is not None else set(params.keys())

    def process_input(self, tool: "BaseTool", kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Inject parameters, enforcing protected ones."""
        result = dict(kwargs)

        for key, value in self._params.items():
            if key in self._enforce:
                # Enforced: always set, overriding any LLM-provided value
                result[key] = value
            elif key not in result:
                # Default: only set if not already provided
                result[key] = value

        return result


class OutputFilterMiddleware(ToolMiddleware):
    """
    Middleware that filters tool output to reduce context bloat.

    When tools return large dicts or objects, this middleware can select
    only the fields that matter, reducing the amount of context passed
    back to the LLM.

    Args:
        allowed_fields: If provided, only these fields will be kept in
            dict outputs. Fields not in this set are removed.
        excluded_fields: If provided, these fields will be removed from
            dict outputs. Cannot be used together with ``allowed_fields``.

    Example:
        >>> middleware = OutputFilterMiddleware(
        ...     allowed_fields={"name", "status", "id"}
        ... )
        >>> tool = FunctionTool.from_defaults(my_fn, middleware=[middleware])

    """

    def __init__(
        self,
        allowed_fields: Optional[Set[str]] = None,
        excluded_fields: Optional[Set[str]] = None,
    ) -> None:
        if allowed_fields is not None and excluded_fields is not None:
            raise ValueError(
                "Cannot specify both allowed_fields and excluded_fields. "
                "Use one or the other."
            )
        self._allowed_fields = allowed_fields
        self._excluded_fields = excluded_fields

    def _filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a dictionary based on allowed/excluded fields."""
        if self._allowed_fields is not None:
            return {k: v for k, v in data.items() if k in self._allowed_fields}
        elif self._excluded_fields is not None:
            return {k: v for k, v in data.items() if k not in self._excluded_fields}
        return data

    def process_output(self, tool: "BaseTool", output: Any) -> Any:
        """Filter output fields."""
        if isinstance(output, dict):
            return self._filter_dict(output)
        elif isinstance(output, list):
            return [
                self._filter_dict(item) if isinstance(item, dict) else item
                for item in output
            ]
        return output
