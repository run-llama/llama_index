"""Make.com API wrapper.

Currently cannot load documents.

"""

from typing import Any, List, Optional
import json

import requests

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document
from gpt_index.response.schema import SourceNode, Response


class MakeWrapper(BaseReader):
    """Make reader."""

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        raise NotImplementedError("Cannot load documents from Make.com API.")

    def pass_response_to_webhook(self, webhook_url: str, response: Response, query: Optional[str] = None) -> str:
        """Pass text to webhook."""
        response_text = response.response
        source_nodes = [n.to_dict() for n in response.source_nodes]
        # source_node_text = response.get_formatted_sources()
        json_dict = {
            "response": response_text,
            "source_nodes": source_nodes,
            "query": query
        }
        r = requests.post(webhook_url, json=json_dict)
        r.raise_for_status()
        print(r)


if __name__ == "__main__":
    wrapper = MakeWrapper()
    test_response = Response(response="test response", source_nodes=[SourceNode(source_text="test source", doc_id="test id")])
    wrapper.pass_response_to_webhook(
        "https://hook.us1.make.com/yjslzjxrthu5p0o2r6bw3nxe5aoepr5g",
        test_response,
        "Test query"
    )
