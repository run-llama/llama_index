"""NeMo Guardrails output parser.

See https://github.com/NVIDIA/NeMo-Guardrails.

"""
try:
    from nemoguardrails import LLMRails, RailsConfig
except ImportError:
    LLMRails = None
    RailsConfig = None

import asyncio
from typing import Any


from llama_index.output_parsers.base import BaseOutputParser


class NeMoGaurdrailsOutputParser(BaseOutputParser):
    """NeMo Gaurdrails output parser."""

    def __init__(self, config: RailsConfig, verbose: bool = False):
        """Initialize a NeMo Guardrails output parser."""
        self.config = config
        self.verbose = verbose
        self.gaurdrails = LLMRails(self.config, verbose=verbose)

    @classmethod
    def from_path(
        cls, config_path: str, verbose: bool = False
    ) -> "NeMoGaurdrailsOutputParser":
        """From file configs."""
        if RailsConfig is None:
            raise ImportError(
                "NeMo Gaurdrails is not installed. Run `pip install nemogaurdrails`. "
                "You may need to upgrade your langchain version after installing."
            )

        config = RailsConfig.from_path(config_path)

        return cls(config, verbose=verbose)

    @classmethod
    def from_content(
        cls, colang_content: str, yaml_content: str, verbose: bool = False
    ) -> "NeMoGaurdrailsOutputParser":
        """From content strings."""
        if RailsConfig is None:
            raise ImportError(
                "NeMo Gaurdrails is not installed. Run `pip install nemogaurdrails`. "
                "You may need to upgrade your langchain version after installing."
            )

        config = RailsConfig.from_content(
            colang_content=colang_content, yaml_content=yaml_content
        )

        return cls(config, verbose=verbose)

    def parse(self, output: str, formatted_prompt: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

        events = [
            {"type": "user_said", "content": formatted_prompt},
            {"type": "bot_said", "content": output},
        ]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "You are using the sync `generate` inside async code. "
                "You should replace with `await generate_async(...)."
            )

        # TODO: super hacky. Is there a better option?
        new_events = asyncio.run(self.gaurdrails.runtime.generate_events(events))
        if len(new_events) >= 2 and new_events[-2].get("type", "") == "bot_said":
            new_output = new_events[-2].get("content", output)
            if not new_output:
                return output

            # TODO use logger here
            if self.verbose:
                print(f"Original:\n{output}\n\nNew:\n{new_output}")

            return new_output

        return output

        # or, another approach, but doesn't seem to work well
        # messages = [
        #    {"type": "user", "content": formatted_prompt},
        #    {"type": "bot", "content": output},
        # ]

        # new_message = self.gaurdrails.generate(messages=messages)

        # return new_message.get("content", output)

    def format(self, query: str) -> str:
        """Unused for NeMoGaurdRailsParser."""
        return query
