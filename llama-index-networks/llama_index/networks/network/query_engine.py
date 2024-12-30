from typing import List, Optional
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.settings import (
    Settings,
)
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.llms import LLM
from llama_index.networks.contributor.query_engine import ContributorQueryEngineClient
import asyncio


class NetworkQueryEngine(BaseQueryEngine):
    """The network Query Engine."""

    def __init__(
        self,
        contributors: List[ContributorQueryEngineClient],
        response_synthesizer: Optional[BaseSynthesizer] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(callback_manager=callback_manager)
        self._contributors = contributors
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=Settings.llm, callback_manager=Settings.callback_manager
        )

    @classmethod
    def from_args(
        cls,
        contributors: List[ContributorQueryEngineClient],
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        summary_template: Optional[BasePromptTemplate] = None,
        simple_template: Optional[BasePromptTemplate] = None,
        output_cls: Optional[BaseModel] = None,
        use_async: bool = False,
        streaming: bool = False,
    ) -> "NetworkQueryEngine":
        llm = llm or Settings.llm

        response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=llm,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            summary_template=summary_template,
            simple_template=simple_template,
            response_mode=response_mode,
            output_cls=output_cls,
            use_async=use_async,
            streaming=streaming,
        )

        callback_manager = Settings.callback_manager

        return cls(
            contributors=contributors,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager,
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {"response_synthesizer": self._response_synthesizer}

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            results = []
            async_tasks = [
                contributor.aquery(query_bundle) for contributor in self._contributors
            ]
            results = run_async_tasks(async_tasks)

            nodes = [
                NodeWithScore(
                    node=TextNode(text=el.response), score=el.metadata["score"]
                )
                for el in results
            ]
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
            # go through all clients
            query_tasks = []
            for contributor in self._contributors:
                query_tasks += [contributor.aquery(query_bundle)]

            results = await asyncio.gather(*query_tasks)
            nodes = [
                NodeWithScore(
                    node=TextNode(text=el.response), score=el.metadata["score"]
                )
                for el in results
            ]
            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @property
    def contributors(self) -> List[ContributorQueryEngineClient]:
        """Get the retriever object."""
        return self._contributors
