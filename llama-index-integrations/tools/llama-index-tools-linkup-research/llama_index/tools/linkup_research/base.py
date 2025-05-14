"""Linkup tool spec."""

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class LinkupToolSpec(BaseToolSpec):
    """Linkup tool spec."""

    spec_functions = [
        "search",
    ]

    def __init__(self, api_key: str, depth: str, output_type: str) -> None:
        """Initialize with parameters."""
        from linkup import LinkupClient

        self.client = LinkupClient(api_key=api_key)
        self.depth = depth
        self.output_type = output_type

    def search(self, query: str):
        """
        Run query through Linkup Search and return metadata.

        Args:
            query: The query to search for.

        """
        api_params = {
            "query": query,
            "depth": self.depth,
            "output_type": self.output_type,
        }
        if self.output_type == "structured":
            if not self.structured_output_schema:
                raise ValueError(
                    "structured_output_schema must be provided when output_type is 'structured'."
                )
            api_params["structured_output_schema"] = self.structured_output_schema
        return self.client.search(**api_params)
