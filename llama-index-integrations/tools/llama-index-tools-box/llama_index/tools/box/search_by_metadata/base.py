from typing import List, Optional
import json
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from box_sdk_gen import (
    BoxClient,
)

from llama_index.readers.box.BoxAPI.box_api import (
    box_check_connection,
    search_files_by_metadata,
    get_box_files_details,
    add_extra_header_to_box_client,
)

from llama_index.readers.box.BoxAPI.box_llama_adaptors import (
    box_file_to_llama_document,
)


class BoxSearchByMetadataOptions:
    """
    Represents options for searching Box resources based on metadata.

    This class provides a way to specify parameters for searching Box resources
    using metadata. You can define the starting point for the search (`from_`), the
    ancestor folder ID to search within (`ancestor_folder_id`), an optional search
    query (`query`), and a limit on the number of returned results (`limit`).

    Attributes:
        from_ (str): The starting point for the search, such as "folder" or "file".
        ancestor_folder_id (str): The ID of the ancestor folder to search within.
        query (Optional[str]): An optional search query string to refine the search
            based on metadata.
        limit (Optional[int]): The maximum number of search results to return.
    """

    from_: str
    ancestor_folder_id: str
    query: Optional[str] = (None,)
    limit: Optional[int] = None

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


class BoxSearchByMetadataToolSpec(BaseToolSpec):
    """
    Provides functionalities for searching Box resources based on metadata.

    This class allows you to search for Box resources based on metadata specified
    using the `BoxSearchByMetadataOptions` class. It utilizes the Box API search
    functionality and returns a list of `Document` objects containing information
    about the found resources.

    Attributes:
        spec_functions (list): A list of supported functions (always "search").
        _box_client (BoxClient): An instance of BoxClient for interacting with Box API.
        _options (BoxSearchByMetadataOptions): An instance of BoxSearchByMetadataOptions
            containing search options.

    Methods:
        search(query_params: Optional[str] = None) -> List[Document]:
            Performs a search for Box resources based on the configured metadata options
            and optional query parameters. Returns a list of `Document` objects representing
            the found resources.
    """

    spec_functions = ["search"]

    _box_client: BoxClient
    _options: BoxSearchByMetadataOptions

    def __init__(
        self, box_client: BoxClient, options: BoxSearchByMetadataOptions
    ) -> None:
        """
        Initializes a `BoxSearchByMetadataToolSpec` instance.

        Args:
            box_client (BoxClient): An authenticated Box API client.
            options (BoxSearchByMetadataToolSpec, optional): An instance of `BoxSearchByMetadataToolSpec` containing search options.
                Defaults to `BoxSearchByMetadataToolSpec()`.
        """
        self._box_client = add_extra_header_to_box_client(box_client)
        self._options = options

    def search(
        self,
        query_params: Optional[str] = None,
    ) -> List[Document]:
        """
        Searches for Box resources based on metadata and returns a list of documents.

        This method leverages the configured metadata options (`self._options`) to
        search for Box resources. It converts the provided JSON string (`query_params`)
        into a dictionary and uses it to refine the search based on additional
        metadata criteria. It retrieves matching Box files and then converts them
        into `Document` objects containing relevant information.

        Args:
            query_params (Optional[str]): An optional JSON string representing additional
                query parameters for filtering by metadata.

        Returns:
            List[Document]: A list of `Document` objects representing the found Box resources.
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
        )

        box_files = get_box_files_details(
            box_client=self._box_client, file_ids=[file.id for file in box_files]
        )

        docs: List[Document] = []

        for file in box_files:
            doc = box_file_to_llama_document(file)
            docs.append(doc)

        return docs
