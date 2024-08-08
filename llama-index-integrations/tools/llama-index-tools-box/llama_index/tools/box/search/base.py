from typing import List, Optional
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from box_sdk_gen import (
    BoxClient,
    SearchForContentScope,
    SearchForContentContentTypes,
)

from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    get_box_files_payload,
    search_files,
)


class BoxSearchOptions:
    scope: Optional[SearchForContentScope] = None
    file_extensions: Optional[List[str]] = None
    created_at_range: Optional[List[str]] = None
    updated_at_range: Optional[List[str]] = None
    size_range: Optional[List[int]] = None
    owner_user_ids: Optional[List[str]] = None
    recent_updater_user_ids: Optional[List[str]] = None
    ancestor_folder_ids: Optional[List[str]] = None
    content_types: Optional[List[SearchForContentContentTypes]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


class BoxSearchToolSpec(BaseToolSpec):
    """Box search tool spec."""

    spec_functions = ["box_search"]

    _box_client: BoxClient
    _options: BoxSearchOptions

    def __init__(
        self, box_client: BoxClient, options: BoxSearchOptions = BoxSearchOptions()
    ) -> None:
        self._box_client = box_client
        self._options = options

    def box_search(
        self,
        query: str,
    ) -> List[Document]:
        """
        Searches for Box resources based on specified criteria and returns a list of their IDs.

        This method utilizes the Box API search functionality to find resources
        matching the provided parameters. It then returns a list containing the IDs
        of the found resources.

        """
        box_check_connection(self._box_client)

        box_files = search_files(
            box_client=self._box_client,
            query=query,
            scope=self._options.scope,
            file_extensions=self._options.file_extensions,
            created_at_range=self._options.created_at_range,
            updated_at_range=self._options.updated_at_range,
            size_range=self._options.size_range,
            owner_user_ids=self._options.owner_user_ids,
            recent_updater_user_ids=self._options.recent_updater_user_ids,
            ancestor_folder_ids=self._options.ancestor_folder_ids,
            content_types=self._options.content_types,
            limit=self._options.limit,
            offset=self._options.offset,
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
