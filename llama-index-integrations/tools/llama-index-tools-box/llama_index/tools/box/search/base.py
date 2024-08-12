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
    """
    Represents options for searching Box resources.

    This class provides a way to specify various criteria for filtering search results
    when using the `BoxSearchToolSpec` class. You can define parameters like search
    scope, file extensions, date ranges (created/updated at), size range, owner IDs,
    and more to refine your search.

    Attributes:
        scope (Optional[SearchForContentScope]): The scope of the search (e.g., all
            content, trashed content).
        file_extensions (Optional[List[str]]): A list of file extensions to filter by.
        created_at_range (Optional[List[str]]): A list representing a date range for
            file creation time (format: YYYY-MM-DD).
        updated_at_range (Optional[List[str]]): A list representing a date range for
            file update time (format: YYYY-MM-DD).
        size_range (Optional[List[int]]): A list representing a range for file size (in bytes).
        owner_user_ids (Optional[List[str]]): A list of user IDs to filter by owner.
        recent_updater_user_ids (Optional[List[str]]): A list of user IDs to filter by
            recent updater.
        ancestor_folder_ids (Optional[List[str]]): A list of folder IDs to search within.
        content_types (Optional[List[SearchForContentContentTypes]]): A list of content
            types to filter by.
        limit (Optional[int]): The maximum number of search results to return.
        offset (Optional[int]): The offset to start results from (for pagination).
    """

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
    """
    Provides functionalities for searching Box resources.

    This class allows you to search for Box resources based on various criteria
    specified using the `BoxSearchOptions` class. It utilizes the Box API search
    functionality and returns a list of `Document` objects containing information
    about the found resources.

    Attributes:
        spec_functions (list): A list of supported functions (always "box_search").
        _box_client (BoxClient): An instance of BoxClient for interacting with Box API.
        _options (BoxSearchOptions): An instance of BoxSearchOptions containing search options.

    Methods:
        box_search(query: str) -> List[Document]:
            Performs a search for Box resources based on the provided query and configured
            search options. Returns a list of `Document` objects representing the found resources.
    """

    spec_functions = ["box_search"]

    _box_client: BoxClient
    _options: BoxSearchOptions

    def __init__(
        self, box_client: BoxClient, options: BoxSearchOptions = BoxSearchOptions()
    ) -> None:
        """
        Initializes a `BoxSearchToolSpec` instance.

        Args:
            box_client (BoxClient): An authenticated Box API client.
            options (BoxSearchOptions, optional): An instance of `BoxSearchOptions` containing search options.
                Defaults to `BoxSearchOptions()`.
        """
        self._box_client = box_client
        self._options = options

    def box_search(
        self,
        query: str,
    ) -> List[Document]:
        """
        Searches for Box resources based on the provided query and configured search options.

        This method utilizes the Box API search functionality to find resources matching the provided
        query and search options specified in the `BoxSearchOptions` object. It returns a list of
        `Document` objects containing information about the found resources.

        Args:
            query (str): The search query to use for searching Box resources.

        Returns:
            List[Document]: A list of `Document` objects representing the found Box resources.
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
