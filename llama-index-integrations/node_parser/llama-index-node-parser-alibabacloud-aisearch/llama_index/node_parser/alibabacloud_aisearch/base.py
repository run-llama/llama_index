import asyncio
import logging
from typing import Any, List, Optional, Sequence

from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.async_utils import run_jobs

from typing import Any, List

from llama_index.core.bridge.pydantic import Field, PrivateAttr

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

try:
    from alibabacloud_searchplat20240529.models import (
        GetDocumentSplitRequest,
        GetDocumentSplitRequestDocument,
        GetDocumentSplitRequestStrategy,
        GetDocumentSplitResponse,
    )
    from alibabacloud_tea_openapi.models import Config as AISearchConfig
    from alibabacloud_searchplat20240529.client import Client
    from Tea.exceptions import TeaException
except ImportError:
    raise ImportError(
        "Could not import alibabacloud_searchplat20240529 python package. "
        "Please install it with `pip install alibabacloud-searchplat20240529`."
    )

logger = logging.getLogger(__name__)


def aretry_decorator(func, wait_seconds: int = 1):
    async def wrap(*args, **kwargs):
        while True:
            try:
                return await func(*args, **kwargs)
            except TeaException as e:
                if e.code == "Throttling.RateQuota":
                    await asyncio.sleep(wait_seconds)
                else:
                    raise

    return wrap


class AlibabaCloudAISearchNodeParser(NodeParser):
    """
    For further details, please visit `https://help.aliyun.com/zh/open-search/search-platform/developer-reference/document-split-api-details`.
    """

    _client: Client = PrivateAttr()
    _split_strategy: GetDocumentSplitRequestStrategy = PrivateAttr()

    image_reader: Optional[BaseReader] = Field(default=None, exclude=True)

    aisearch_api_key: str = Field(default=None, exclude=True)
    endpoint: str = None

    service_id: str = "ops-document-split-001"
    workspace_name: str = "default"
    chunk_size: int = 300
    need_sentence: bool = False
    default_content_encoding: str = "utf8"
    default_content_type: str = "text/plain"
    num_workers: int = 4

    def __init__(
        self, endpoint: str = None, aisearch_api_key: str = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.aisearch_api_key = get_from_param_or_env(
            "aisearch_api_key", aisearch_api_key, "AISEARCH_API_KEY"
        )
        self.endpoint = get_from_param_or_env("endpoint", endpoint, "AISEARCH_ENDPOINT")

        config = AISearchConfig(
            bearer_token=self.aisearch_api_key,
            endpoint=self.endpoint,
            protocol="http",
        )

        self._client = Client(config=config)
        self._split_strategy = GetDocumentSplitRequestStrategy(
            max_chunk_size=self.chunk_size, need_sentence=self.need_sentence
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AlibabaCloudAISearchNodeParser"

    def _parse_nodes(
        self,
        documents: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        return asyncio.get_event_loop().run_until_complete(
            self._aparse_nodes(documents, show_progress, **kwargs)
        )

    async def _aparse_nodes(
        self,
        documents: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Parse document into nodes.

        Args:
            nodes (Sequence[BaseNode]): nodes to parse

        """
        jobs = [self._aparse_node(d) for d in documents]
        results = await run_jobs(
            jobs,
            workers=self.num_workers,
            desc="Parsing documents into nodes",
            show_progress=show_progress,
        )
        # return flattened results
        return [item for sublist in results for item in sublist]

    @aretry_decorator
    async def _aparse_node(
        self,
        node: BaseNode,
    ) -> List[BaseNode]:
        content_type = getattr(node, "mimetype", self.default_content_type)
        main_type, sub_type = content_type.split("/")
        if main_type != "text":
            raise ValueError(f"Unsupported content type: {content_type}")
        content_encoding = node.metadata.get("encoding", self.default_content_encoding)
        document = GetDocumentSplitRequestDocument(
            content=node.get_content(),
            content_encoding=content_encoding,
            content_type=sub_type,
        )
        request = GetDocumentSplitRequest(
            document=document, strategy=self._split_strategy
        )

        response: GetDocumentSplitResponse = (
            await self._client.get_document_split_async(
                workspace_name=self.workspace_name,
                service_id=self.service_id,
                request=request,
            )
        )
        return await self._handle_response(response, node)

    async def _handle_response(
        self, response: GetDocumentSplitResponse, node: BaseNode
    ) -> List[TextNode]:
        chunks = list(response.body.result.chunks)
        if response.body.result.sentences:
            chunks.extend(response.body.result.sentences)
        chunks.extend(await self._handle_rich_texts(response.body.result.rich_texts))
        return build_nodes_from_splits(
            [chunk.content for chunk in chunks], node, id_func=self.id_func
        )

    async def _handle_rich_texts(self, rich_texts) -> List[str]:
        chunks = []
        if not rich_texts:
            return chunks
        chunks = list(rich_texts)
        for chunk in chunks:
            if chunk.meta.get("type") == "image":
                chunk.content = await self._handle_image(chunk.content)
        return chunks

    async def _handle_image(self, url: str) -> str:
        content = url
        if not self.image_reader:
            return content
        try:
            docs = await self.image_reader.aload_data([url])
            content = docs[0].get_content()
        except Exception as e:
            logger.error(f"Read image {url} error: {e}")
        return content
