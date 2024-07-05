import asyncio
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers import Refine
from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow
from llama_index.core.workflow.decorators import step

# pip install llama-index-llms-openai
from llama_index.llms.openai import OpenAI


@dataclass
class QueryEvent(Event):
    query: str


@dataclass
class RetrieverEvent(Event):
    nodes: list[NodeWithScore]


@dataclass
class QueryResult(Event):
    nodes: list[NodeWithScore]


class RAGWorkflow(Workflow):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Shared state: to be refactored into a better workflow context manager
        self.index = None
        self.query: str | None = None

    @step(StartEvent)
    async def ingest(self, ev: StartEvent):
        dsname = ev.get("dataset")
        if not dsname:
            return None

        _, documents = download_llama_dataset(dsname, "./data")
        self.index = VectorStoreIndex.from_documents(documents=documents)
        return StopEvent(msg=f"Indexed {len(documents)} documents.")

    @step(StartEvent)
    async def retrieve(self, ev: StartEvent):
        query = ev.get("query")
        if not query:
            return None

        self.query = query
        print(f"Query the database with: {self.query}")
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

    @step(RetrieverEvent)
    async def rerank(self, ev: RetrieverEvent):
        ranker = LLMRerank(choice_batch_size=5, top_n=3)
        new_nodes = ranker.postprocess_nodes(ev.nodes, query_str=self.query)
        print(f"Reranked nodes to {len(new_nodes)}")
        return QueryResult(nodes=new_nodes)

    @step(QueryResult)
    async def synthesize(self, ev: QueryResult):
        # Should never fallback, it'll get better once we have a proper context storage
        query = self.query or ""

        llm = OpenAI(model="gpt-3.5-turbo")
        summarizer = Refine(llm=llm, verbose=True)
        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(msg=str(response))


async def main():
    w = RAGWorkflow(timeout=10)

    print("Ingesting data...")
    ret = await w.run(dataset="PaulGrahamEssayDataset")
    print(ret)

    print("Querying...")
    ret = await w.run(query="Who is Paul Graham?")
    print(ret)


if __name__ == "__main__":
    asyncio.run(main())
