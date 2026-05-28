from typing import Any, List, Optional, Sequence, Union

from pydantic import AnyUrl

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    ImageBlock,
    TextBlock,
    VideoBlock,
)
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.base import BaseGPTIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter, TextSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate, RichPromptTemplate
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.schema import (
    MediaResource,
    MetadataMode,
    Node,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.settings import Settings

CITATION_QA_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

CITATION_REFINE_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. "
    "We have provided an existing answer: {existing_answer}"
    "Below are several numbered sources of information. "
    "Use them to refine the existing answer. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "\nBegin refining!"
    "\n------\n"
    "{context_msg}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

CITATION_CHAT_CONTENT_QA_TEMPLATE = RichPromptTemplate("""
{% chat role="user" %}
Please provide an answer based solely on the provided sources. When referencing information from a source, cite the appropriate source(s) using their corresponding numbers. Every answer should include at least one source citation. Only cite a source when you are explicitly referencing it. If none of the sources are helpful, you should indicate that. For example:
Source 1:
The sky is red in the evening and blue in the morning.
Source 2:
Water is wet when the sky is red.
Query: When is water wet?
Answer: Water will be wet when the sky is red [2], which occurs in the evening [1].
Now it's your turn. Below are several numbered sources of information:
------
{% for message in context_messages %}
{% for block in message.blocks %}
{% if block.block_type == 'text' %}
{{ block.text }}
{% elif block.block_type == 'image' %}
{{ block.inline_url() | image }}
{% elif block.block_type == 'audio' %}
{{ block.inline_url() | audio }}
{% elif block.block_type == 'video' %}
{{ block.inline_url() | video }}
{% endif %}

{% endfor %}
{% endfor %}
------
Query: {{ query_str }}
Answer:
{% endchat %}
""")

CITATION_CHAT_CONTENT_REFINE_TEMPLATE = RichPromptTemplate("""
{% chat role="user" %}
Please provide an answer based solely on the provided sources. When referencing information from a source, cite the appropriate source(s) using their corresponding numbers. Every answer should include at least one source citation. Only cite a source when you are explicitly referencing it. If none of the sources are helpful, you should indicate that. For example:
Source 1:
The sky is red in the evening and blue in the morning.
Source 2:
Water is wet when the sky is red.
Query: When is water wet?
Answer: Water will be wet when the sky is red [2], which occurs in the evening [1].
Now it's your turn. We have provided an existing answer: {{ existing_answer }}
Below are several numbered sources of information. Use them to refine the existing answer. If the provided sources are not helpful, you will repeat the existing answer.
Begin refining!
------
{% for message in context_messages %}
{% for block in message.blocks %}
{% if block.block_type == 'text' %}
{{ block.text }}
{% elif block.block_type == 'image' %}
{{ block.inline_url() | image }}
{% elif block.block_type == 'audio' %}
{{ block.inline_url() | audio }}
{% elif block.block_type == 'video' %}
{{ block.inline_url() | video }}
{% endif %}

{% endfor %}
{% endfor %}
------
Query: {{ query_str }}
Answer:
{% endchat %}
""")

DEFAULT_CITATION_CHUNK_SIZE = 512
DEFAULT_CITATION_CHUNK_OVERLAP = 20


class CitationQueryEngine(BaseQueryEngine):
    """
    Citation query engine.

    Args:
        retriever (BaseRetriever): A retriever object.
        response_synthesizer (Optional[BaseSynthesizer]):
            A BaseSynthesizer object.
        citation_chunk_size (int):
            Size of citation chunks, default=512. Useful for controlling
            granularity of sources.
        citation_chunk_overlap (int): Overlap of citation nodes, default=20.
        text_splitter (Optional[TextSplitter]):
            A text splitter for creating citation source nodes. Default is
            a SentenceSplitter.
        callback_manager (Optional[CallbackManager]): A callback manager.
        metadata_mode (MetadataMode): A MetadataMode object that controls how
            metadata is included in the citation prompt.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        citation_chunk_size: int = DEFAULT_CITATION_CHUNK_SIZE,
        citation_chunk_overlap: int = DEFAULT_CITATION_CHUNK_OVERLAP,
        text_splitter: Optional[TextSplitter] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
        multimodal: bool = False,
    ) -> None:
        self.text_splitter = text_splitter or SentenceSplitter(
            chunk_size=citation_chunk_size, chunk_overlap=citation_chunk_overlap
        )
        self._citation_chunk_size = citation_chunk_size
        self._citation_chunk_overlap = citation_chunk_overlap
        self._multimodal = multimodal
        self._retriever = retriever

        callback_manager = callback_manager or Settings.callback_manager
        llm = llm or Settings.llm

        if response_synthesizer is None:
            synthesizer_kwargs: dict = {
                "llm": llm,
                "callback_manager": callback_manager,
                "text_qa_template": CITATION_QA_TEMPLATE,
                "refine_template": CITATION_REFINE_TEMPLATE,
                "response_mode": ResponseMode.COMPACT,
                "use_async": False,
                "streaming": False,
            }
            if multimodal:
                synthesizer_kwargs["chat_content_qa_template"] = (
                    CITATION_CHAT_CONTENT_QA_TEMPLATE
                )
                synthesizer_kwargs["chat_content_refine_template"] = (
                    CITATION_CHAT_CONTENT_REFINE_TEMPLATE
                )
                synthesizer_kwargs["multimodal"] = True
            self._response_synthesizer = get_response_synthesizer(**synthesizer_kwargs)
        else:
            self._response_synthesizer = response_synthesizer

        self._node_postprocessors = node_postprocessors or []
        self._metadata_mode = metadata_mode

        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = callback_manager

        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_args(
        cls,
        index: BaseGPTIndex,
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        citation_chunk_size: int = DEFAULT_CITATION_CHUNK_SIZE,
        citation_chunk_overlap: int = DEFAULT_CITATION_CHUNK_OVERLAP,
        text_splitter: Optional[TextSplitter] = None,
        citation_qa_template: BasePromptTemplate = CITATION_QA_TEMPLATE,
        citation_refine_template: BasePromptTemplate = CITATION_REFINE_TEMPLATE,
        citation_chat_content_qa_template: BasePromptTemplate = CITATION_CHAT_CONTENT_QA_TEMPLATE,
        citation_chat_content_refine_template: BasePromptTemplate = CITATION_CHAT_CONTENT_REFINE_TEMPLATE,
        retriever: Optional[BaseRetriever] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        use_async: bool = False,
        streaming: bool = False,
        # class-specific args
        metadata_mode: MetadataMode = MetadataMode.NONE,
        multimodal: bool = False,
        **kwargs: Any,
    ) -> "CitationQueryEngine":
        """
        Initialize a CitationQueryEngine object.".

        Args:
            index: (BastGPTIndex): index to use for querying
            llm: (Optional[LLM]): LLM object to use for response generation.
            citation_chunk_size (int):
                Size of citation chunks, default=512. Useful for controlling
                granularity of sources.
            citation_chunk_overlap (int): Overlap of citation nodes, default=20.
            text_splitter (Optional[TextSplitter]):
                A text splitter for creating citation source nodes. Default is
                a SentenceSplitter.
            citation_qa_template (BasePromptTemplate): Template for initial citation QA
            citation_refine_template (BasePromptTemplate):
                Template for citation refinement.
            retriever (BaseRetriever): A retriever object.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
                node postprocessors.
            verbose (bool): Whether to print out debug info.
            response_mode (ResponseMode): A ResponseMode object.
            use_async (bool): Whether to use async.
            streaming (bool): Whether to use streaming.
            optimizer (Optional[BaseTokenUsageOptimizer]): A BaseTokenUsageOptimizer
                object.

        """
        retriever = retriever or index.as_retriever(**kwargs)

        if response_synthesizer is None:
            synthesizer_kwargs: dict = {
                "llm": llm,
                "text_qa_template": citation_qa_template,
                "refine_template": citation_refine_template,
                "response_mode": response_mode,
                "use_async": use_async,
                "streaming": streaming,
            }
            if multimodal:
                synthesizer_kwargs["chat_content_qa_template"] = (
                    citation_chat_content_qa_template
                )
                synthesizer_kwargs["chat_content_refine_template"] = (
                    citation_chat_content_refine_template
                )
                synthesizer_kwargs["multimodal"] = True
            response_synthesizer = get_response_synthesizer(**synthesizer_kwargs)

        return cls(
            retriever=retriever,
            llm=llm,
            response_synthesizer=response_synthesizer,
            callback_manager=Settings.callback_manager,
            citation_chunk_size=citation_chunk_size,
            citation_chunk_overlap=citation_chunk_overlap,
            text_splitter=text_splitter,
            node_postprocessors=node_postprocessors,
            metadata_mode=metadata_mode,
            multimodal=multimodal,
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {"response_synthesizer": self._response_synthesizer}

    def _create_citation_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Modify retrieved nodes to be granular sources."""
        if self._multimodal:
            return self._create_multimodal_citation_nodes(nodes)

        new_nodes: List[NodeWithScore] = []
        for node in nodes:
            text_chunks = self.text_splitter.split_text(
                node.node.get_content(metadata_mode=self._metadata_mode)
            )

            for text_chunk in text_chunks:
                text = f"Source {len(new_nodes) + 1}:\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node.model_dump()),
                    score=node.score,
                )
                new_node.node.set_content(text)
                new_nodes.append(new_node)
        return new_nodes

    @staticmethod
    def _coerce_url(url: Union[AnyUrl, str, None]) -> Optional[AnyUrl]:
        if url is None:
            return None
        if isinstance(url, AnyUrl):
            return url
        return AnyUrl(str(url))

    def _create_multimodal_citation_nodes(
        self, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Split multimodal retrieved nodes into per-source citation nodes."""
        new_nodes: List[NodeWithScore] = []
        for node in nodes:
            blocks = node.node.get_content_blocks(metadata_mode=self._metadata_mode)
            chat_msg = ChatMessage(blocks=blocks)
            split_msgs = chat_msg.split(
                max_tokens=self._citation_chunk_size,
                overlap=self._citation_chunk_overlap,
            )

            for split_msg in split_msgs:
                source_label = f"Source {len(new_nodes) + 1}:"
                new_node = Node.model_validate(
                    {
                        "embedding": node.node.embedding,
                        "metadata": dict(node.node.metadata),
                        "excluded_embed_metadata_keys": list(
                            node.node.excluded_embed_metadata_keys
                        ),
                        "excluded_llm_metadata_keys": list(
                            node.node.excluded_llm_metadata_keys
                        ),
                        "relationships": dict(node.node.relationships),
                    }
                )

                text_parts: List[str] = [source_label]
                for blk in split_msg.blocks:
                    if isinstance(blk, TextBlock):
                        text_parts.append(blk.text)
                    elif isinstance(blk, ImageBlock):
                        new_node.image_resource = MediaResource(
                            data=blk.image if isinstance(blk.image, bytes) else None,
                            url=self._coerce_url(blk.url),
                            path=blk.path,
                            mimetype=blk.image_mimetype,
                        )
                    elif isinstance(blk, AudioBlock):
                        new_node.audio_resource = MediaResource(
                            data=blk.audio if isinstance(blk.audio, bytes) else None,
                            url=self._coerce_url(blk.url),
                            path=blk.path,
                            mimetype=blk.format,
                        )
                    elif isinstance(blk, VideoBlock):
                        new_node.video_resource = MediaResource(
                            data=blk.video if isinstance(blk.video, bytes) else None,
                            url=self._coerce_url(blk.url),
                            path=blk.path,
                            mimetype=blk.video_mimetype,
                        )

                new_node.text_resource = MediaResource(
                    text="\n".join(text_parts) + "\n"
                )
                new_nodes.append(NodeWithScore(node=new_node, score=node.score))
        return new_nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)

        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

        return nodes

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)

        for postprocessor in self._node_postprocessors:
            nodes = await postprocessor.apostprocess_nodes(
                nodes, query_bundle=query_bundle
            )

        return nodes

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        nodes = self._create_citation_nodes(nodes)
        return self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        nodes = self._create_citation_nodes(nodes)
        return await self._response_synthesizer.asynthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self.retrieve(query_bundle)
                nodes = self._create_citation_nodes(nodes)

                retrieve_event.on_end(payload={EventPayload.NODES: nodes})

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self.aretrieve(query_bundle)
                nodes = self._create_citation_nodes(nodes)

                retrieve_event.on_end(payload={EventPayload.NODES: nodes})

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
