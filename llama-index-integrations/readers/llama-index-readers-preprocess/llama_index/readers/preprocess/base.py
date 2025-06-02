"""Preprocess Reader."""

import hashlib
import os
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import (
    Document,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)


class PreprocessReader(BaseReader):
    """
    Preprocess is an API service that splits any kind of document into optimal chunks of text for use in language model tasks.
    Preprocess splits the documents into chunks of text that respect the layout and semantics of the original document.
    Learn more at https://preprocess.co/.

    Args:
        api_key (str):
            [Required] The Preprocess API Key.
            If you don't have one yet, please request it at support@preprocess.co.
            Default: `None`

        file_path (str):
            [Required] The path to the document to be preprocessed (convertend and split into chunks).
            Default: `None`

        table_output_format (str):
            The output format for tables within the document.
            Accepted values are [text, markdown, html].
            Default: `text`

        repeat_table_header (bool):
            If `True`, when tables are split across multiple chunks, each chunk will include the table's row header.
            Default: `False`

        merge (bool):
            If `True`, short chunks will be merged with others to maximize chunk length.
            Default: `False`

        repeat_title (bool):
            If `True`, each chunk will include the title of the parent paragraph or section.
            Default: `False`

        keep_header (bool):
            If `True`, the content of each page's header will be included in the chunks.
            Default: `True`

        smart_header (bool):
            If `True`, only relevant headers will be included in the chunks, while irrelevant information will be removed.
            Relevant headers are those that serve as section or paragraph titles.
            If set to `False`, only the `keep_header` parameter will be considered. If `keep_header` is `False`, this parameter will be ignored.
            Default: `True`

        keep_footer (bool):
            If `True`, the content of each page's footer will be included in the chunks.
            Default: `False`

        image_text (bool):
            If `True`, the text contained in images will be added to the chunks.
            Default: `False`


    Examples:
        >>> loader = PreprocessReader(api_key="your-api-key", file_path="valid/path/to/file")

    """

    def __init__(self, api_key: str, *args, **kwargs):
        """Initialise with parameters."""
        try:
            from pypreprocess import Preprocess
        except ImportError:
            raise ImportError(
                "`pypreprocess` package not found, please run `pip install"
                " pypreprocess`"
            )

        if api_key is None or api_key == "":
            raise ValueError(
                "Please provide an api key to be used while doing the auth with the system."
            )
        _info = {}
        self._preprocess = Preprocess(api_key)
        self._filepath = None
        self._process_id = None

        for key, value in kwargs.items():
            if key == "filepath":
                self._filepath = value
                self._preprocess.set_filepath(value)

            if key == "process_id":
                self._process_id = value
                self._preprocess.set_process_id(value)

            elif key in ["table_output_format", "table_output"]:
                _info["table_output_format"] = value

            elif key in ["repeat_table_header", "table_header"]:
                _info["repeat_table_header"] = value

            elif key in [
                "merge",
                "repeat_title",
                "keep_header",
                "keep_footer",
                "smart_header",
                "image_text",
            ]:
                _info[key] = value

        if _info != {}:
            self._preprocess.set_info(_info)

        if self._filepath is None and self._process_id is None:
            raise ValueError(
                "Please provide either filepath or process_id to handle the resutls."
            )

        self._chunks = None

    def load_data(self, return_whole_document=False) -> List[Document]:
        """
        Load data from Preprocess.

        Args:
            return_whole_document (bool):
                Returning a list of one element, that element containing the full document.
                Default: `false`

        Examples:
            >>> documents = loader.load_data()
            >>> documents = loader.load_data(return_whole_document=True)

        Returns:
            List[Document]:
                A list of documents each document containing a chunk from the original document.

        """
        if self._chunks is None:
            if self._process_id is not None:
                self._get_data_by_process()
            elif self._filepath is not None:
                self._get_data_by_filepath()

            if self._chunks is not None:
                if return_whole_document is True:
                    return [
                        Document(
                            text=" ".join(self._chunks),
                            metadata={"filename": os.path.basename(self._filepath)},
                        )
                    ]
                else:
                    return [
                        Document(
                            text=chunk,
                            metadata={"filename": os.path.basename(self._filepath)},
                        )
                        for chunk in self._chunks
                    ]
            else:
                raise Exception(
                    "There is error happened during handling your file, please try again."
                )

        else:
            if return_whole_document is True:
                return [
                    Document(
                        text=" ".join(self._chunks),
                        metadata={"filename": os.path.basename(self._filepath)},
                    )
                ]
            else:
                return [
                    Document(
                        text=chunk,
                        metadata={"filename": os.path.basename(self._filepath)},
                    )
                    for chunk in self._chunks
                ]

    def get_process_id(self):
        """
        Get process's hash id from Preprocess.

        Examples:
            >>> process_id = loader.get_process_id()

        Returns:
            str:
                Process's hash id.

        """
        return self._process_id

    def get_nodes(self) -> List[TextNode]:
        """
        Get nodes from Preprocess's chunks.

        Examples:
            >>> nodes = loader.get_nodes()

        Returns:
            List[TextNode]:
                List of nodes, each node will contains a chunk from the original document.

        """
        if self._chunks is None:
            self.load_data()

        nodes = []
        for chunk in self._chunks:
            text = str(chunk)
            id = hashlib.md5(text.encode()).hexdigest()
            nodes.append(TextNode(text=text, id_=id))

        if len(nodes) > 1:
            nodes[0].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[1].node_id,
                metadata={"filename": os.path.basename(self._filepath)},
            )
            for i in range(1, len(nodes) - 1):
                nodes[i].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=nodes[i + 1].node_id,
                    metadata={"filename": os.path.basename(self._filepath)},
                )
                nodes[i].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=nodes[i - 1].node_id,
                    metadata={"filename": os.path.basename(self._filepath)},
                )

            nodes[-1].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[-2].node_id,
                metadata={"filename": os.path.basename(self._filepath)},
            )
        return nodes

    def _get_data_by_filepath(self) -> None:
        pp_response = self._preprocess.chunk()
        if pp_response.status == "OK" and pp_response.success is True:
            self._process_id = pp_response.data["process"]["id"]
            response = self._preprocess.wait()
            if response.status == "OK" and response.success is True:
                # self._filepath = response.data['info']['file']['name']
                self._chunks = response.data["chunks"]

    def _get_data_by_process(self) -> None:
        response = self._preprocess.wait()
        if response.status == "OK" and response.success is True:
            self._filepath = response.data["info"]["file"]["name"]
            self._chunks = response.data["chunks"]
