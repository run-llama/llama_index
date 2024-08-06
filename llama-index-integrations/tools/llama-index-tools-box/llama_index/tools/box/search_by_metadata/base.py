from typing import Dict, List, Optional
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from box_sdk_gen import (
    BoxClient,
)

from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_payload,
    search_files_by_metadata,
)


class BoxSearchByMetadataToolSpec(BaseToolSpec):
    """Box search tool spec."""

    _box_client: BoxClient

    def __init__(self, box_client: BoxClient) -> None:
        self._box_client = box_client

    def search(
        self,
        from_: str,
        ancestor_folder_id: str,
        query: Optional[str] = None,
        query_params: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        marker: Optional[str] = None,
    ) -> List[Document]:
        """
        Searches for Box resources based on metadata and returns a list of Llama Index
        Documents.

        This method utilizes the Box API search functionality to find resources
        matching the provided metadata query. It then returns a list containing the IDs
        of the found resources.

        Args:
            box_client (BoxClient): An authenticated Box client object used
                for interacting with the Box API.
            from_ (str): The metadata template key to search from.
            ancestor_folder_id (str): The ID of the Box folder to search within.
            query (Optional[str], optional): A search query string. Defaults to None.
            query_params (Optional[Dict[str, str]], optional): Additional query parameters
                to filter the search results. Defaults to None.
            limit (Optional[int], optional): The maximum number of results to return.
                Defaults to None.
            marker (Optional[str], optional): The marker for the start of the next page of
                results. Defaults to None.

        Returns:
            List[str]: A list of Box resource IDs matching the search criteria.
        """
        box_check_connection(self._box_client)
        box_files = search_files_by_metadata(
            box_client=self._box_client,
            from_=from_,
            ancestor_folder_id=ancestor_folder_id,
            query=query,
            query_params=query_params,
            limit=limit,
            marker=marker,
        )
        box_payloads = get_box_files_payload(
            self._box_client, [box_file.id for box_file in box_files]
        )

        docs: List[Document] = []

        for box_payload in box_payloads:
            file = box_payload.resource_info
            doc = Document(
                extra_info=file.to_dict(),
                metadata=file.to_dict(),
            )
            docs.append(doc)

        return docs
