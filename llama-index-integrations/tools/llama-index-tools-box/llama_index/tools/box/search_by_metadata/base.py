from typing import List, Optional
import json
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


class BoxSearchByMetadataOptions:
    from_: str
    ancestor_folder_id: str
    query: Optional[str] = (None,)
    # query_params: Optional[Dict[str, str]] = (None,)
    limit: Optional[int] = None
    # marker: Optional[str] = None # The AI agent won't know what to do with this...

    def __init__(
        self,
        from_: str,
        ancestor_folder_id: str,
        query: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.from_ = from_
        self.ancestor_folder_id = ancestor_folder_id
        self.query = query
        self.limit = limit
        # self.marker = marker # The AI agent won't know what to do with this...


class BoxSearchByMetadataToolSpec(BaseToolSpec):
    """Box search tool spec."""

    spec_functions = ["search"]

    _box_client: BoxClient
    _options: BoxSearchByMetadataOptions

    def __init__(
        self, box_client: BoxClient, options: BoxSearchByMetadataOptions
    ) -> None:
        self._box_client = box_client
        self._options = options

    def search(
        self,
        # query: Optional[str] = None,
        # query_params: Optional[Dict[str, str]] = None,
        query_params: Optional[str] = None,
    ) -> List[Document]:
        """
        Searches for Box resources based on metadata and returns a list of Llama Index
        Documents.

        This method utilizes the Box API search functionality to find resources
        matching the provided metadata query. It then returns a list containing the IDs
        of the found resources.
        """
        box_check_connection(self._box_client)

        # Box API accepts a dictionary of query parameters as a string, so we need to
        # convert the provided JSON string to a dictionary.
        params_dict = json.loads(query_params)

        box_files = search_files_by_metadata(
            box_client=self._box_client,
            from_=self._options.from_,
            ancestor_folder_id=self._options.ancestor_folder_id,
            query=self._options.query,
            query_params=params_dict,
            limit=self._options.limit,
            # marker=self._options.marker, # The AI agent won't know what to do with this...
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
