import asyncio
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.moss import MossToolSpec, QueryOptions
from inferedge_moss import MossClient, DocumentInfo


async def main():
    # 1. Initialize Client
    MOSS_PROJECT_KEY = os.getenv("MOSS_PROJECT_KEY")
    MOSS_PROJECT_ID = os.getenv("MOSS_PROJECT_ID")
    client = MossClient(project_id=MOSS_PROJECT_ID, project_key=MOSS_PROJECT_KEY)

    # 2. Configure query settings (optional — defaults: top_k=5, alpha=0.5, model_id="moss-minilm")
    query_options = QueryOptions(top_k=5, alpha=0.5, model_id="moss-minilm")

    # 3. Initialize Tool
    moss_tool = MossToolSpec(
        client=client,
        index_name="knowledge_base_new",
        query_options=query_options,
    )

    # 4. List existing indexes before indexing
    print("\n[Step 4] Listing existing indexes...")
    print(await moss_tool.list_indexes())

    # 5. Index Documents
    print("\n[Step 5] Indexing Documents...")
    docs: List[DocumentInfo] = [
        DocumentInfo(
            id="123",
            text="LlamaIndex is a data framework for LLM-based applications.",
            metadata={"source": "docs", "category": "framework"},
        ),
        DocumentInfo(
            id="124",
            text="Moss is a real-time semantic search engine optimized for speed.",
            metadata={"source": "moss_website", "category": "engine"},
        ),
    ]
    await moss_tool.index_docs(docs)
    print(f"Indexed {len(docs)} documents.")

    # 6. List indexes again to confirm creation
    print("\n[Step 6] Listing indexes after indexing...")
    print(await moss_tool.list_indexes())

    # 7. Create agent with all exposed tools (query, list_indexes, delete_index)
    print("\n[Step 7] Creating Agent...")
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    agent = ReActAgent(
        tools=moss_tool.to_tool_list(),
        llm=llm,
        verbose=True,
    )

    # 8. Run Agent — natural language query triggers the query tool
    print("\n[Step 8] Querying via Agent...")
    response = await agent.run(user_msg="What is Moss?")
    print("\nAgent Response:")
    print(response)

    # 9. Run Agent — ask it to list available indexes
    print("\n[Step 9] Listing indexes via Agent...")
    response = await agent.run(user_msg="What indexes are available?")
    print("\nAgent Response:")
    print(response)

    # 10. Clean up — delete the index directly (not via agent to avoid accidental deletion)
    print("\n[Step 10] Cleaning up...")
    print(await moss_tool.delete_index("knowledge_base"))


if __name__ == "__main__":
    asyncio.run(main())
