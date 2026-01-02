from __future__ import annotations

from typing import Any, Dict, Optional

import google.genai.types as types


class GenerationConfigBuilder:
    """Build and merge generation configuration for Gemini requests."""

    def __init__(
        self,
        *,
        base_generation_config: Dict[str, Any],
        temperature: float,
        max_output_tokens: Optional[int],
        cached_content: Optional[str],
        built_in_tool: Optional[types.Tool],
    ) -> None:
        self._base_generation_config = base_generation_config
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._cached_content = cached_content
        self._built_in_tool = built_in_tool

    def build(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a merged generation config dictionary."""
        config: Dict[str, Any] = {**(self._base_generation_config or {})}

        config.setdefault("temperature", self._temperature)
        if self._max_output_tokens is not None:
            config.setdefault("max_output_tokens", self._max_output_tokens)
        if self._cached_content is not None:
            config.setdefault("cached_content", self._cached_content)

        if self._built_in_tool is not None:
            tools = config.get("tools")
            if tools is None:
                tools = []
                config["tools"] = tools
            if isinstance(tools, list):
                if tools:
                    raise ValueError(
                        "Providing multiple Google GenAI tools or mixing with custom tools is not supported."
                    )
                tools.append(self._built_in_tool)

        if overrides:
            config.update(overrides)

        return config
