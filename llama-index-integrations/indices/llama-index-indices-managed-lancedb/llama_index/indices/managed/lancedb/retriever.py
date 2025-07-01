import os
from lancedb.table import Table, AsyncTable
from PIL import Image
from dataclasses import dataclass

from .utils import query_multimodal, query_text, aquery_multimodal, aquery_text
from llama_index.core.llms import ImageBlock
from llama_index.core.schema import ImageDocument
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from typing import Union, Optional, List, Any
from typing_extensions import override


@dataclass
class ExtendedQueryBundle(QueryBundle):
    image: Optional[Union[Image.Image, ImageBlock, ImageDocument, str]] = None


class LanceDBRetriever(BaseRetriever):
    def __init__(
        self, table: Union[AsyncTable, Table], multimodal: bool, **kwargs: Any
    ):
        self.table = table
        self.multimodal = multimodal
        callback_manager = kwargs.get("callback_manager")
        verbose = kwargs.get("verbose", False)
        super().__init__(callback_manager, verbose)

    def _retrieve(self, query_bundle: ExtendedQueryBundle) -> List[NodeWithScore]:
        if not self.multimodal:
            return query_text(table=self.table, query=query_bundle.query_str)
        else:
            if not query_bundle.image and not query_bundle.image_path:
                raise ValueError(
                    "No image or image_path has been provided, but retrieval is set to multi-modal."
                )
            elif query_bundle.image:
                return query_multimodal(table=self.table, query=query_bundle.image)
            elif query_bundle.image_path:
                img = ImageBlock(path=query_bundle.image_path)
                return query_multimodal(table=self.table, query=img)
            else:
                return []

    async def _aretrieve(
        self, query_bundle: ExtendedQueryBundle
    ) -> List[NodeWithScore]:
        if not self.multimodal:
            return await aquery_text(table=self.table, query=query_bundle.query_str)
        else:
            if not query_bundle.image and not query_bundle.image_path:
                raise ValueError(
                    "No image or image_path has been provided, but retrieval is set to multi-modal."
                )
            elif query_bundle.image:
                return await aquery_multimodal(
                    table=self.table, query=query_bundle.image
                )
            elif query_bundle.image_path:
                img = ImageBlock(path=query_bundle.image_path)
                return await aquery_multimodal(table=self.table, query=img)
            else:
                return []

    @override
    def retrieve(
        self,
        query_str: Optional[str] = None,
        query_image: Optional[
            Union[Image.Image, ImageBlock, ImageDocument, str]
        ] = None,
        query_image_path: Optional[os.PathLike[str]] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieves nodes relevant to the given query.

        Args:
            query_str (Optional[str]): The text query string. Required if the retriever is not multimodal.
            query_image (Optional[Union[Image.Image, ImageBlock, ImageDocument, str]]): The image query, which can be a PIL Image, ImageBlock, ImageDocument, or a string path/URL. Used if the retriever is multimodal.
            query_image_path (Optional[os.PathLike[str]]): The file path to the image query. Used if the retriever is multimodal.

        Returns:
            List[NodeWithScore]: A list of nodes with associated relevance scores.

        Raises:
            ValueError: If none of the query parameters are provided.
            ValueError: If a text query is not provided for a non-multimodal retriever.
            ValueError: If neither an image nor image path is provided for a multimodal retriever.

        """
        if not query_str and not query_image and not query_image_path:
            raise ValueError(
                "At least one among query_str, query_image and query_image_path needs to be set"
            )
        if not self.multimodal:
            if query_str:
                query_bundle = ExtendedQueryBundle(query_str=query_str)
            else:
                raise ValueError(
                    "No query_str provided, but the retriever is not multimodal"
                )
        else:
            if query_image:
                query_bundle = ExtendedQueryBundle(query_str="", image=query_image)
            elif query_image_path:
                query_bundle = ExtendedQueryBundle(
                    query_str="", image_path=query_image_path
                )
            else:
                raise ValueError(
                    "No query_image or query_image_path provided, but the retriever is multimodal"
                )

        return self._retrieve(query_bundle=query_bundle)

    @override
    async def aretrieve(
        self,
        query_str: Optional[str] = None,
        query_image: Optional[
            Union[Image.Image, ImageBlock, ImageDocument, str]
        ] = None,
        query_image_path: Optional[os.PathLike[str]] = None,
    ) -> List[NodeWithScore]:
        """
        Asynchronously retrieves nodes relevant to the given query.

        Args:
            query_str (Optional[str]): The text query string. Required if the retriever is not multimodal.
            query_image (Optional[Union[Image.Image, ImageBlock, ImageDocument, str]]): The image query, which can be a PIL Image, ImageBlock, ImageDocument, or a string path/URL. Used if the retriever is multimodal.
            query_image_path (Optional[os.PathLike[str]]): The file path to the image query. Used if the retriever is multimodal.

        Returns:
            List[NodeWithScore]: A list of nodes with associated relevance scores.

        Raises:
            ValueError: If none of the query parameters are provided.
            ValueError: If a text query is not provided for a non-multimodal retriever.
            ValueError: If neither an image nor image path is provided for a multimodal retriever.

        """
        if not query_str and not query_image and not query_image_path:
            raise ValueError(
                "At least one among query_str, query_image and query_image_path needs to be set"
            )
        if not self.multimodal:
            if query_str:
                query_bundle = ExtendedQueryBundle(query_str=query_str)
            else:
                raise ValueError(
                    "No query_str provided, but the retriever is not multimodal"
                )
        else:
            if query_image:
                query_bundle = ExtendedQueryBundle(query_str="", image=query_image)
            elif query_image_path:
                query_bundle = ExtendedQueryBundle(
                    query_str="", image_path=query_image_path
                )
            else:
                raise ValueError(
                    "No query_image or query_image_path provided, but the retriever is multimodal"
                )
        return await self._aretrieve(query_bundle=query_bundle)
