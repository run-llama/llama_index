"""
LlamaIndex tools for APIVerve.

Access 300+ utility APIs for AI agents including validation, conversion,
generation, analysis, and lookup tools.

Example:
    >>> from llama_index.tools.apiverve import APIVerveToolSpec
    >>> from llama_index.agent.openai import OpenAIAgent
    >>>
    >>> apiverve = APIVerveToolSpec(api_key="your-api-key")
    >>> agent = OpenAIAgent.from_tools(apiverve.to_tool_list())
    >>> response = agent.chat("Is test@example.com a valid email?")

For more information, see: https://docs.apiverve.com
"""

from llama_index.tools.apiverve.base import APIVerveToolSpec

__all__ = ["APIVerveToolSpec"]
