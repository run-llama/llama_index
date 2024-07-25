import asyncio
from dataclasses import dataclass
from typing import Optional, List, Any, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers import Refine
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow
from llama_index.core.workflow.decorators import step
from llama_index.core.workflow.drawing import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)

# pip install llama-index-llms-openai
from llama_index.llms.openai import OpenAI


@dataclass
class QueryEvent(Event):
    query: str


@dataclass
class RetrieverEvent(Event):
    nodes: List[NodeWithScore]


@dataclass
class QueryResult(Event):
    nodes: List[NodeWithScore]


class RAGWorkflow(Workflow):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Shared state: to be refactored into a better workflow context manager
        self.index: Any = None
        self.query: str = ""

    @step()
    async def ingest(self, ev: StartEvent) -> Optional[StopEvent]:
        dsname = ev.get("dataset")
        if not dsname:
            return None

        _, documents = download_llama_dataset(dsname, "./data")
        self.index = VectorStoreIndex.from_documents(documents=documents)
        return StopEvent(msg=f"Indexed {len(documents)} documents.")

    @step()
    async def retrieve(self, ev: StartEvent) -> Optional[RetrieverEvent]:
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")
        if self.index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step()
    async def rerank(
        self, ev: Union[RetrieverEvent, StartEvent]
    ) -> Optional[QueryResult]:
        if isinstance(ev, StartEvent):
            self.query = ev.get("query", "")
            return None
        elif isinstance(ev, RetrieverEvent):
            ranker = LLMRerank(choice_batch_size=5, top_n=3)
            new_nodes = ranker.postprocess_nodes(ev.nodes, query_str=self.query)
            print(f"Reranked nodes to {len(new_nodes)}")
            return QueryResult(nodes=new_nodes)
        else:
            return None

    @step()
    async def synthesize(
        self, ev: Union[QueryResult, StartEvent]
    ) -> Optional[StopEvent]:
        # Should never fallback, it'll get better once we have a proper context storage
        if isinstance(ev, StartEvent):
            self.query = ev.get("query", "")
            return None
        elif isinstance(ev, QueryResult):
            llm = OpenAI(model="gpt-3.5-turbo")
            summarizer = Refine(llm=llm, verbose=True)
            response = await summarizer.asynthesize(self.query, nodes=ev.nodes)
            return StopEvent(msg=str(response))
        else:
            return None


async def main() -> None:
    w = RAGWorkflow(timeout=60, verbose=True)

    print("Ingesting data...")
    ret = await w.run(dataset="PaulGrahamEssayDataset")
    print(ret)

    print("Querying...")
    ret = await w.run(query="Who is Paul Graham?")
    print(ret)

    draw_all_possible_flows(w)
    draw_most_recent_execution(w)


if __name__ == "__main__":
    asyncio.run(main())
