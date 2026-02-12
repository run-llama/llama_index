import httpx
import asyncio
from uuid import uuid4
from typing import List, Optional, Union
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.llms import ChatMessage


# ================= Event Definitions =================
class GradeEvent(Event):
    """Retrieval complete, awaiting grading."""

    nodes: List[NodeWithScore]
    query: str


class RetryRequestEvent(Event):
    """Grading failed, requesting retry (intermediate state)."""

    original_query: str
    feedback: str


class RewriteEvent(Event):
    """Rewrite complete, carrying new Query (used for retrieval)."""

    original_query: str
    feedback: str


class WebSearchEvent(Event):
    """Local retries exhausted, switching to web search."""

    query: str


class GenerateEvent(Event):
    """Grading passed, preparing generation."""

    nodes: List[NodeWithScore]
    source: str


# ================= Workflow Definition =================
class EduMatrixWorkflow(Workflow):
    def __init__(
        self, retriever, llm, timeout: int = 60, tavily_api_key: Optional[str] = None
    ):
        super().__init__(timeout=timeout)
        self.retriever = retriever
        self.llm = llm

        # 1. Save Key for manual implementation
        self.tavily_api_key = tavily_api_key
        # 2. Toggle logic: Enable search only if Key is provided
        self.enable_web_search = bool(tavily_api_key)

        self._http_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    # --- HTTP Client Management (Lazy Load) ---
    async def _get_client(self) -> httpx.AsyncClient:
        # Return existing client if available and open
        if self._http_client is not None and not self._http_client.is_closed:
            return self._http_client

        # Lock to ensure thread-safe initialization
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(timeout=10.0)
            return self._http_client

    async def aclose(self):
        """Clean up resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    # --- Search Logic (Manual Implementation) ---
    async def _tavily_search(self, query: str) -> List[NodeWithScore]:
        if not self.enable_web_search or not self.tavily_api_key:
            return []

        try:
            client = await self._get_client()
            resp = await client.post(
                url="https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_api_key,  # Explicitly use stored key
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,
                    "max_results": 3,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            nodes = []
            # Add Direct Answer if available
            if data.get("answer"):
                nodes.append(
                    NodeWithScore(
                        node=TextNode(
                            text=f"[Web Summary]: {data['answer']}",
                            metadata={"file_name": "Web", "source": "web"},
                        ),
                        score=0.9,
                    )
                )

            # Add Search Results
            for res in data.get("results", []):
                nodes.append(
                    NodeWithScore(
                        node=TextNode(
                            text=f"{res['content']}\n(Source: {res['url']})",
                            metadata={"file_name": "Web", "source": "web"},
                        ),
                        score=0.8,
                    )
                )
            return nodes
        except Exception as e:
            print(f"❌ Web Search Error: {e}")
            return []

    # --- Step 1: Retrieve ---
    @step
    async def retrieve(
        self, ctx: Context, ev: Union[StartEvent, RewriteEvent]
    ) -> GradeEvent:
        trace_id = await ctx.store.get("trace_id", default=uuid4().hex[:8])

        if isinstance(ev, StartEvent):
            question = ev.get("question")
            if not question:
                print(f"[{trace_id}] ⚠️ No question provided.")
                return GradeEvent(nodes=[], query="")

            await ctx.store.set("trace_id", trace_id)
            await ctx.store.set("original_question", question)
            await ctx.store.set("retry_count", 0)
            search_query = question
            print(f"[{trace_id}] 🚀 Start: {search_query}")
        else:
            # RewriteEvent
            search_query = ev.original_query
            print(f"[{trace_id}] 🔄 Rewritten Retrieval: {search_query}")

        if not self.retriever:
            return GradeEvent(nodes=[], query=search_query)

        nodes = await self.retriever.aretrieve(search_query)
        return GradeEvent(nodes=nodes[:10], query=search_query)

    # --- Step 2: Grade ---
    @step
    async def grade(
        self, ctx: Context, ev: GradeEvent
    ) -> Union[GenerateEvent, RetryRequestEvent, WebSearchEvent]:
        trace_id = await ctx.store.get("trace_id")
        nodes = ev.nodes

        # Empty retrieval check
        if not nodes:
            return await self._handle_retry(ctx, ev.query, "No content")

        preview = "\n".join([n.node.get_content()[:200] for n in nodes[:5]])

        # LLM Grader
        prompt = (
            f"Query: {ev.query}\nContext: {preview}\n"
            "Does this context contain ANY information that could help answer the query? "
            "Reply only 'yes' or 'no'."
        )
        res = await self.llm.acomplete(prompt)
        score_raw = res.text.strip().lower()
        is_relevant = "yes" in score_raw

        if is_relevant:
            print(f"[{trace_id}] ✅ Grade Pass")
            return GenerateEvent(nodes=nodes, source="local")

        print(f"[{trace_id}] ❌ Grade Fail: {score_raw}")
        return await self._handle_retry(ctx, ev.query, "Irrelevant content")

    async def _handle_retry(self, ctx: Context, query: str, reason: str):
        MAX_RETRIES = 1
        retry_count = await ctx.store.get("retry_count", default=0)

        if retry_count < MAX_RETRIES:
            await ctx.store.set("retry_count", retry_count + 1)
            return RetryRequestEvent(original_query=query, feedback=reason)

        # Fallback Logic
        original_q = await ctx.store.get("original_question")
        if self.enable_web_search:
            print(f"[{await ctx.store.get('trace_id')}] 🌍 Switching to Web Search...")
            return WebSearchEvent(query=original_q)
        else:
            print(
                f"[{await ctx.store.get('trace_id')}] ⚠️ Web Search Disabled. Forcing Generation."
            )
            # Fallback to generation (best effort) if web search is disabled
            return GenerateEvent(nodes=[], source="fallback")

    # --- Step 3: Rewrite ---
    @step
    async def rewrite(self, ctx: Context, ev: RetryRequestEvent) -> RewriteEvent:
        trace_id = await ctx.store.get("trace_id")
        print(f"[{trace_id}] 🧠 Rewriting query...")

        prompt = (
            f"The original query '{ev.original_query}' failed to retrieve relevant info.\n"
            f"Please extract core entities and generate a new search keyword.\n"
            f"Output only the keyword."
        )
        res = await self.llm.acomplete(prompt)
        new_q = res.text.strip()

        return RewriteEvent(original_query=new_q, feedback="refined")

    # --- Step 4: Web Search ---
    @step
    async def web_search(self, ctx: Context, ev: WebSearchEvent) -> GenerateEvent:
        # Directly call the manual Tavily implementation
        nodes = await self._tavily_search(ev.query)
        return GenerateEvent(nodes=nodes, source="web")

    # --- Step 5: Generate ---
    @step
    async def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent:
        nodes = ev.nodes
        original_q = await ctx.store.get("original_question")

        serialized_nodes = []
        context_lines = []

        for n in nodes[:8]:
            meta = n.node.metadata or {}
            text = n.node.get_content()
            citation = (
                "[Web]"
                if ev.source == "web"
                else f"[{meta.get('file_name', 'Doc')} P{meta.get('page_label', '?')}]"
            )
            context_lines.append(f"Citation {citation}:\n{text}\n")

            serialized_nodes.append(
                {
                    "id": n.node.node_id,
                    "text": text,
                    "metadata": meta,
                    "score": n.score,
                    "source": ev.source,
                }
            )

        if not serialized_nodes and ev.source != "fallback":
            return StopEvent(
                result={
                    "final_response": "No relevant information found.",
                    "retrieved_nodes": [],
                }
            )

        sys_msg = ChatMessage(
            role="system", content="Answer based on the provided context. Cite sources."
        )
        user_msg = ChatMessage(
            role="user",
            content=f"Context:\n{''.join(context_lines)}\n\nQuestion: {original_q}",
        )

        stream = await self.llm.astream_chat([sys_msg, user_msg])

        return StopEvent(
            result={"final_response": stream, "retrieved_nodes": serialized_nodes}
        )
