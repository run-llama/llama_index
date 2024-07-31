import asyncio
from typing import Optional, List, Any, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers import Refine
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    draw_all_possible_flows,
    draw_most_recent_execution,
)
from llama_index.core.workflow.context import Context

# pip install llama-index-llms-ollama
from llama_index.llms.ollama import Ollama


class QueryEvent(Event):
    query: str


class RetrieverEvent(Event):
    nodes: List[NodeWithScore]


class QueryResult(Event):
    nodes: List[NodeWithScore]


class RAGWorkflow(Workflow):
    @step(pass_context=True)
    async def ingest(self, ctx: Context, ev: StartEvent) -> Optional[StopEvent]:
        dsname = ev.get("dataset")
        if not dsname:
            return None

        _, documents = download_llama_dataset(dsname, "./data")
        ctx["INDEX"] = VectorStoreIndex.from_documents(documents=documents)
        return StopEvent(result=f"Indexed {len(documents)} documents.")

    @step(pass_context=True)
    async def retrieve(self, ctx: Context, ev: StartEvent) -> Optional[RetrieverEvent]:
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        index: Any = ctx.get("INDEX")
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
        )
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step(pass_context=True)
    def rerank(
        self, ctx: Context, ev: Union[RetrieverEvent, StartEvent]
    ) -> Optional[QueryResult]:
        if isinstance(ev, StartEvent):
            ctx["QUERY"] = ev.get("query", "")
            return None
        elif isinstance(ev, RetrieverEvent):
            ranker = LLMRerank(choice_batch_size=5, top_n=3)
            new_nodes = ranker.postprocess_nodes(ev.nodes, query_str=ctx.get("QUERY"))
            print(f"Reranked nodes to {len(new_nodes)}")
            return QueryResult(nodes=new_nodes)
        else:
            return None


@step(workflow=RAGWorkflow, pass_context=True)
async def synthesize(
    ctx: Context, ev: Union[QueryResult, StartEvent]
) -> Optional[StopEvent]:
    # Should never fallback, it'll get better once we have a proper context storage
    if isinstance(ev, StartEvent):
        ctx["QUERY"] = ev.get("query", "")
        return None
    elif isinstance(ev, QueryResult):
        llm = Ollama(model="llama3.1:8b", request_timeout=120)
        summarizer = Refine(llm=llm, streaming=True, verbose=True)
        query = ctx.get("QUERY", "")
        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)
    else:
        return None


async def main() -> None:
    w = RAGWorkflow(timeout=60, verbose=True)

    print("Ingesting data...")
    ret = await w.run(dataset="PaulGrahamEssayDataset")
    print(ret)

    print("Querying...")
    ret = await w.run(query="Who is Paul Graham?")
    async for chunk in ret.async_response_gen():
        print(chunk, end="", flush=True)

    draw_all_possible_flows(w)
    draw_most_recent_execution(w)


if __name__ == "__main__":
    asyncio.run(main())
