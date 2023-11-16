"""Guardrails output parser.

See https://github.com/ShreyaR/guardrails.

"""
from deprecated import deprecated

try:
    from guardrails import Guard
except ImportError:
    Guard = None
    PromptCallable = None

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from llama_index.bridge.langchain import BaseLLM
from llama_index.types import BaseOutputParser


def get_callable(llm: Optional["BaseLLM"]) -> Optional[Callable]:
    """Get callable."""
    if llm is None:
        return None

    return llm.__call__


class GuardrailsOutputParser(BaseOutputParser):
    """Guardrails output parser."""

    def __init__(
        self,
        guard: Guard,
        llm: Optional["BaseLLM"] = None,
        format_key: Optional[str] = None,
    ):
        """Initialize a Guardrails output parser."""
        self.guard: Guard = guard
        self.llm = llm
        self.format_key = format_key

    @classmethod
    @deprecated(version="0.8.46")
    def from_rail(
        cls, rail: str, llm: Optional["BaseLLM"] = None
    ) -> "GuardrailsOutputParser":
        """From rail."""
        if Guard is None:
            raise ImportError(
                "Guardrails is not installed. Run `pip install guardrails-ai`. "
            )

        return cls(Guard.from_rail(rail), llm=llm)

    @classmethod
    @deprecated(version="0.8.46")
    def from_rail_string(
        cls, rail_string: str, llm: Optional["BaseLLM"] = None
    ) -> "GuardrailsOutputParser":
        """From rail string."""
        if Guard is None:
            raise ImportError(
                "Guardrails is not installed. Run `pip install guardrails-ai`. "
            )

        return cls(Guard.from_rail_string(rail_string), llm=llm)

    def parse(
        self,
        output: str,
        llm: Optional["BaseLLM"] = None,
        num_reasks: Optional[int] = 1,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Parse, validate, and correct errors programmatically."""
        llm = llm or self.llm
        llm_fn = get_callable(llm)

        return self.guard.parse(
            output, llm_api=llm_fn, num_reasks=num_reasks, *args, **kwargs
        )

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        output_schema_text = deepcopy(self.guard.rail.prompt)

        # Add format instructions here.
        format_instructions_tmpl = self.guard.raw_prompt.format_instructions
        # NOTE: output_schema is fixed
        format_instructions = format_instructions_tmpl.format(
            output_schema=output_schema_text
        )

        if self.format_key is not None:
            fmt_query = query.format(**{self.format_key: format_instructions})
        else:
            fmt_query = query + "\n\n" + format_instructions

        return fmt_query
