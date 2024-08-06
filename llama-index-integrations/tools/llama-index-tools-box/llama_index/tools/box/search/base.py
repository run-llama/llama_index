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


class BoxSearchToolSpec(BaseToolSpec):
    """Box search tool spec."""

    _box_client: BoxClient

    def __init__(self, box_client: BoxClient) -> None:
        self._box_client = box_client

    def search(
        self,
        query: Optional[str] = None,
        scope: Optional[SearchForContentScope] = None,
        file_extensions: Optional[List[str]] = None,
        created_at_range: Optional[List[str]] = None,
        updated_at_range: Optional[List[str]] = None,
        size_range: Optional[List[int]] = None,
        owner_user_ids: Optional[List[str]] = None,
        recent_updater_user_ids: Optional[List[str]] = None,
        ancestor_folder_ids: Optional[List[str]] = None,
        content_types: Optional[List[SearchForContentContentTypes]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Document]:
        """
        Searches for Box resources based on specified criteria and returns a list of their IDs.

        This method utilizes the Box API search functionality to find resources
        matching the provided parameters. It then returns a list containing the IDs
        of the found resources.

        Args:
            query (Optional[str], optional): A search query string. Defaults to None.
            scope (Optional[SearchForContentScope], optional): The scope of the search.
                Defaults to None.
            file_extensions (Optional[List[str]], optional): A list of file extensions
                to filter by. Defaults to None.
            created_at_range (Optional[List[str]], optional): A list representing a date
                range for file creation time. Defaults to None.
            updated_at_range (Optional[List[str]], optional): A list representing a date
                range for file update time. Defaults to None.
            size_range (Optional[List[int]], optional): A list representing a size range
                for files. Defaults to None.
            owner_user_ids (Optional[List[str]], optional): A list of user IDs to filter
                by owner. Defaults to None.
            recent_updater_user_ids (Optional[List[str]], optional): A list of user IDs to
                filter by recent updater. Defaults to None.
            ancestor_folder_ids (Optional[List[str]], optional): A list of folder IDs to
                search within. Defaults to None.
            content_types (Optional[List[SearchForContentContentTypes]], optional): A list
                of content types to filter by. Defaults to None.
            limit (Optional[int], optional): The maximum number of results to return.
                Defaults to None.
            offset (Optional[int], optional): The number of results to skip before
                starting to collect. Defaults to None.

        Returns:
            List[Document]: A list of documents representing the found resources.
        """
        box_check_connection(self._box_client)
        box_files = search_files(
            box_client=self._box_client,
            query=query,
            scope=scope,
            file_extensions=file_extensions,
            created_at_range=created_at_range,
            updated_at_range=updated_at_range,
            size_range=size_range,
            owner_user_ids=owner_user_ids,
            recent_updater_user_ids=recent_updater_user_ids,
            ancestor_folder_ids=ancestor_folder_ids,
            content_types=content_types,
            limit=limit,
            offset=offset,
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
