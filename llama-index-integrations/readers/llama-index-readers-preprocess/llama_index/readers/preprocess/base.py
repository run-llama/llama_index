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
    def __init__(self, api_key: str, *args, **kwargs):
        if api_key is None or api_key == "":
            raise ValueError(
                "Please provide an api key to be used while doing the auth with the system."
            )

        try:
            from pypreprocess import Preprocess
        except ImportError:
            raise ImportError(
                "`pypreprocess` package not found, please run `pip install"
                " pypreprocess`"
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
        return self._process_id

    def get_nodes(self) -> List[TextNode]:
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
