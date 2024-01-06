"""Make.com API wrapper.

Currently cannot load documents.

"""

from typing import Any, List, Optional

import requests

from llama_index.core.response.schema import Response
from llama_index.readers.base import BaseReader
from llama_index.schema import Document, NodeWithScore, TextNode


class MakeWrapper(BaseReader):
    """Make reader."""

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        NOTE: This is not implemented.

        """
        raise NotImplementedError("Cannot load documents from Make.com API.")

    def pass_response_to_webhook(
        self, webhook_url: str, response: Response, query: Optional[str] = None
    ) -> None:
        """Pass response object to webhook.

        Args:
            webhook_url (str): Webhook URL.
            response (Response): Response object.
            query (Optional[str]): Query. Defaults to None.

        """
        response_text = response.response
        source_nodes = [n.dict() for n in response.source_nodes]
        json_dict = {
            "response": response_text,
            "source_nodes": source_nodes,
            "query": query,
        }
        r = requests.post(webhook_url, json=json_dict)
        r.raise_for_status()


if __name__ == "__main__":
    wrapper = MakeWrapper()
    test_response = Response(
        response="test response",
        source_nodes=[NodeWithScore(node=TextNode(text="test source", id_="test id"))],
    )
    wrapper.pass_response_to_webhook(
        "https://hook.us1.make.com/asdfadsfasdfasdfd",
        test_response,
        "Test query",
    )
