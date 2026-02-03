import asyncio
import os
from typing import List

from llama_index.core.agent import ReActAgent
from llama_index.core.llms import MockLLM
from llama_index.tools.moss import MossToolSpec
from inferedge_moss import MossClient, DocumentInfo


async def main():
    print("--- Moss Tool with ReAct Agent Example ---\n")

    # 1. Initialize Client
    # Ensure you have your environment variables set if needed by MossClient
    # or pass credentials directly.
    client = MossClient()

    # 2. Initialize Tool
    print("Initializing MossToolSpec...")
    moss_tool = MossToolSpec(
        client=client,
        index_name="knowledge_base",
        top_k=3,
        alpha=0.5
    )

    # 3. Index Documents (Optional step)
    print("\n[Step 1] Indexing Documents...")
    docs = [
        DocumentInfo(
            text="LlamaIndex is a data framework for LLM-based applications.",
            metadata={"source": "docs", "category": "framework"}
        ),
        DocumentInfo(
            text="Moss is a real-time semantic search engine optimized for speed.",
            metadata={"source": "moss_website", "category": "engine"}
        ),
    ]
    await moss_tool.index_docs(docs)
    print(f"Indexed {len(docs)} documents.")

    # 4. Create Agent
    # using MockLLM for demonstration; replace with OpenAI() or similar in production
    print("\n[Step 2] Creating Agent...")
    llm = MockLLM()
    agent = ReActAgent.from_tools(
        moss_tool.to_tool_list(),
        llm=llm,
        verbose=True
    )

    # 5. Run Agent
    print("\n[Step 3] Querying...")
    # This query would trigger the tool usage in a real scenario
    response = await agent.achat("What is Moss?")
    print("\nAgent Response:")
    print(response)


if __name__ == "__main__":
    # Ensure we catch ImportError for better user experience if run without deps
    try:
        asyncio.run(main())
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required dependencies: pip install llama-index-tools-moss llama-index-core inferedge-moss")
    except Exception as e:
        print(f"An error occurred: {e}")
