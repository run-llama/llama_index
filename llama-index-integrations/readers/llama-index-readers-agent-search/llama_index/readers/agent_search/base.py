r"""
AgentSearch reader.

Example as of 1/8/2024:

```python
AgentSearch = download_loader("AgentSearch")

document = reader.load_data(
    query="latest news",
    search_provider="bing"
)[0]

print(f'Document:\n{document} ')
```

```plaintext
Document:
Doc ID: 67a57dfe-8bd6-4c69-af9d-683e76177119
Text: The latest news encompasses a wide array of topics, reflecting
the dynamic and complex nature of the world today. Notable events
include the conviction of a man for killing his ex-wife's new partner,
highlighting the ongoing issue of domestic violence and its legal
consequences [2]. In the realm of international relations, the release
of Jeffrey...
```

For more information, refer to the docs:
https://agent-search.readthedocs.io/en/latest/
"""

from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class AgentSearchReader(BaseReader):
    """AgentSearch reader."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize with parameters."""
        import_err_msg = (
            "`agent-search` package not found, please run `pip install agent-search`"
        )
        try:
            import agent_search  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        from agent_search import SciPhi

        self._client = SciPhi(api_base=api_base, api_key=api_key)

    def load_data(
        self,
        query: str,
        search_provider: str = "bing",
        llm_model: str = "SciPhi/Sensei-7B-V1",
    ) -> List[Document]:
        """
        Load data from AgentSearch, hosted by SciPhi.

        Args:
            collection_name (str): Name of the Milvus collection.
            query_vector (List[float]): Query vector.
            limit (int): Number of results to return.

        Returns:
            List[Document]: A list of documents.

        """
        rag_response = self._client.get_search_rag_response(
            query=query, search_provider=search_provider, llm_model=llm_model
        )
        return [Document(text=rag_response.pop("response"), metadata=rag_response)]
