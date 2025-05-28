"""
Unstructured file reader.

A parser for unstructured text files using Unstructured.io.
Supports .csv, .tsv, .doc, .docx, .odt, .epub, .org, .rst, .rtf,
.md, .msg, .pdf, .heic, .png, .jpg, .jpeg, .tiff, .bmp, .ppt, .pptx,
.xlsx, .eml, .html, .xml, .txt, .json documents.

"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, NodeRelationship, TextNode

try:
    from unstructured.documents.elements import Element
except ImportError:
    Element = None


class UnstructuredReader(BaseReader):
    """General unstructured text reader for a variety of files."""

    def __init__(
        self,
        *args: Any,
        api_key: str = None,
        url: str = None,
        allowed_metadata_types: Optional[Tuple] = None,
        excluded_metadata_keys: Optional[Set] = None,
    ) -> None:
        """
        Initialize UnstructuredReader.

        Args:
            *args (Any): Additional arguments passed to the BaseReader.
            api_key (str, optional): API key for accessing the Unstructured.io API. If provided, the reader will use the API for parsing files. Defaults to None.
            url (str, optional): URL for the Unstructured.io API. If not provided and an api_key is given, defaults to "http://localhost:8000". Ignored if api_key is not provided. Defaults to None.
            allowed_metadata_types (Optional[Tuple], optional): Tuple of types that are allowed in the metadata. Defaults to (str, int, float, type(None)).
            excluded_metadata_keys (Optional[Set], optional): Set of metadata keys to exclude from the final document. Defaults to {"orig_elements"}.

        Attributes:
            api_key (str or None): Stores the API key.
            use_api (bool): Indicates whether to use the API for parsing files, based on the presence of the api_key.
            url (str or None): URL for the Unstructured.io API if using the API.
            allowed_metadata_types (Tuple): Tuple of types that are allowed in the metadata.
            excluded_metadata_keys (Set): Set of metadata keys to exclude from the final document.

        """
        super().__init__(*args)  # not passing kwargs to parent bc it cannot accept it

        if Element is None:
            raise ImportError(
                "Unstructured is not installed. Please install it using 'pip install -U unstructured'."
            )

        self.api_key = api_key
        self.use_api = bool(api_key)
        self.url = url or "http://localhost:8000" if self.use_api else None
        self.allowed_metadata_types = allowed_metadata_types or (
            str,
            int,
            float,
            type(None),
        )
        self.excluded_metadata_keys = excluded_metadata_keys or {"orig_elements"}

    @classmethod
    def from_api(cls, api_key: str, url: str = None):
        """Set the server url and api key."""
        return cls(api_key, url)

    def load_data(
        self,
        file: Optional[Path] = None,
        unstructured_kwargs: Optional[Dict] = None,
        document_kwargs: Optional[Dict] = None,
        extra_info: Optional[Dict] = None,
        split_documents: Optional[bool] = False,
        excluded_metadata_keys: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load data using Unstructured.io.

        Depending on the configuration, if url is set or use_api is True,
        it'll parse the file using an API call, otherwise it parses it locally.
        extra_info is extended by the returned metadata if split_documents is True.

        Args:
            file (Optional[Path]): Path to the file to be loaded.
            unstructured_kwargs (Optional[Dict]): Additional arguments for unstructured partitioning.
            document_kwargs (Optional[Dict]): Additional arguments for document creation.
            extra_info (Optional[Dict]): Extra information to add to the document metadata.
            split_documents (Optional[bool]): Whether to split the documents.
            excluded_metadata_keys (Optional[List[str]]): Keys to exclude from the metadata.

        Returns:
            List[Document]: List of parsed documents.

        """
        unstructured_kwargs = unstructured_kwargs.copy() if unstructured_kwargs else {}

        if (
            unstructured_kwargs.get("file") is not None
            and unstructured_kwargs.get("metadata_filename") is None
        ):
            raise ValueError(
                "Please provide a 'metadata_filename' as part of the 'unstructured_kwargs' when loading a file stream."
            )

        elements: List[Element] = self._partition_elements(unstructured_kwargs, file)

        return self._create_documents(
            elements,
            document_kwargs,
            extra_info,
            split_documents,
            excluded_metadata_keys,
        )

    def _partition_elements(
        self, unstructured_kwargs: Dict, file: Optional[Path] = None
    ) -> List[Element]:
        """
        Partition the elements from the file or via API.

        Args:
            file (Optional[Path]): Path to the file to be loaded.
            unstructured_kwargs (Dict): Additional arguments for unstructured partitioning.

        Returns:
            List[Element]: List of partitioned elements.

        """
        if file:
            unstructured_kwargs["filename"] = str(file)

        if self.use_api:
            from unstructured.partition.api import partition_via_api

            return partition_via_api(
                api_key=self.api_key,
                api_url=self.url + "/general/v0/general",
                **unstructured_kwargs,
            )
        else:
            from unstructured.partition.auto import partition

            return partition(**unstructured_kwargs)

    def _create_documents(
        self,
        elements: List[Element],
        document_kwargs: Optional[Dict],
        extra_info: Optional[Dict],
        split_documents: Optional[bool],
        excluded_metadata_keys: Optional[List[str]],
    ) -> List[Document]:
        """
        Create documents from partitioned elements.

        Args:
            elements (List): List of partitioned elements.
            document_kwargs (Optional[Dict]): Additional arguments for document creation.
            extra_info (Optional[Dict]): Extra information to add to the document metadata.
            split_documents (Optional[bool]): Whether to split the documents.
            excluded_metadata_keys (Optional[List[str]]): Keys to exclude from the metadata.

        Returns:
            List[Document]: List of parsed documents.

        """
        doc_kwargs = document_kwargs or {}
        doc_extras = extra_info or {}
        excluded_keys = set(excluded_metadata_keys or self.excluded_metadata_keys)
        docs: List[Document] = []

        def _merge_metadata(
            element: Element, sequence_number: Optional[int] = None
        ) -> Dict[str, Any]:
            candidate_metadata = {**element.metadata.to_dict(), **doc_extras}
            metadata = {
                key: (
                    value
                    if isinstance(value, self.allowed_metadata_types)
                    else json.dumps(value)
                )
                for key, value in candidate_metadata.items()
                if key not in excluded_keys
            }
            if sequence_number is not None:
                metadata["sequence_number"] = sequence_number
            return metadata

        if len(elements) == 0:
            return []

        text_chunks = [" ".join(str(el).split()) for el in elements]
        metadata = _merge_metadata(elements[0])
        filename = metadata.get("file_path", None) or metadata["filename"]
        source = Document(
            text="\n\n".join(text_chunks),
            extra_info=metadata,
            doc_id=filename,
            id_=filename,
            **doc_kwargs,
        )

        if split_documents:
            docs = []
            for sequence_number, element in enumerate(elements):
                hash_id = element.id_to_hash(sequence_number)
                node = TextNode(
                    text=element.text,
                    metadata=_merge_metadata(element, sequence_number),
                    doc_id=hash_id,
                    id_=hash_id,
                    **doc_kwargs,
                )
                node.relationships[NodeRelationship.SOURCE] = (
                    source.as_related_node_info()
                )
                docs.append(node)
        else:
            docs = [source]

        return docs
