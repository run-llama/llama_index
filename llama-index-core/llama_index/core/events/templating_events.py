"""Events for Templating."""

from dataclasses import dataclass
from typing import Dict, Optional

from deprecated import deprecated

from llama_index.core.callbacks.schema import CBEvent, CBEventType
from llama_index.core.prompts.base import BasePromptTemplate


@dataclass
class TemplatingStartEventPayload:
    """Payload for TemplatingStartEvent."""

    template: str
    template_vars: Dict[str, str]
    system_prompt: str | None
    query_wrapper_prompt: Optional[BasePromptTemplate]


class TemplatingStartEvent(CBEvent):
    """Event to indicate templating has started."""

    template: str
    template_vars: Dict[str, str]
    system_prompt: str | None
    query_wrapper_prompt: Optional[BasePromptTemplate]

    def __init__(
        self,
        template: str,
        template_vars: Dict[str, str],
        system_prompt: str | None,
        query_wrapper_prompt: Optional[BasePromptTemplate],
    ):
        """Initialize the event."""
        super().__init__(event_type=CBEventType.TEMPLATING)
        self.template = template
        self.template_vars = template_vars
        self.system_prompt = system_prompt
        self.query_wrapper_prompt = query_wrapper_prompt

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> TemplatingStartEventPayload:
        """Return the payload for the event."""
        return TemplatingStartEventPayload(
            template=self.template,
            template_vars=self.template_vars,
            system_prompt=self.system_prompt,
            query_wrapper_prompt=self.query_wrapper_prompt,
        )


@dataclass
class TemplatingEndEventPayload:
    """Payload for TemplatingEndEvent."""


class TemplatingEndEvent(CBEvent):
    """Event to indicate templating has ended."""

    @property
    @deprecated("You can access the payload properties directly from the class")
    def payload(self) -> TemplatingEndEventPayload:
        """Return the payload for the event."""
        return TemplatingEndEventPayload()
