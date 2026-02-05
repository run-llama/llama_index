import asyncio
import os
from typing import List

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.moss import MossToolSpec, QueryOptions
from inferedge_moss import MossClient, DocumentInfo


async def main():
    print("--- Moss Tool with ReAct Agent Example ---\n")

    # 1. Initialize Client
    # Ensure you have your environment variables set or pass credentials directly.
    MOSS_PROJECT_KEY = os.getenv('MOSS_PROJECT_KEY')
    MOSS_PROJECT_ID = os.getenv('MOSS_PROJECT_ID')
    client = MossClient(project_id=MOSS_PROJECT_ID, project_key=MOSS_PROJECT_KEY)
    # 2. Configure query settings - Instantiate QueryOptions (Optional)
    # If skipped, the tool will use its own defaults.
    query_options = QueryOptions(top_k=12, alpha=0.9)
    # 3. Initialize Tool
    print("Initializing MossToolSpec...")
    moss_tool = MossToolSpec(
        client=client,
        index_name="knowledge_base",
        query_options=query_options
    )

    # 4. Index Documents (Optional step)
    print("\n[Step 4] Indexing Documents...")
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

    # 5. Create an agent (Using OpenAI llm for demonstration)
    print("\n[Step 5] Creating Agent...")
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-here')
    llm = OpenAI()
    agent = ReActAgent.from_tools(
        moss_tool.to_tool_list(),
        llm=llm,
        verbose=True
    )

    # 6. Run Agent
    print("\n[Step 6] Querying...")
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
        print("Please install required dependencies: pip install llama-index-tools-moss llama-index-core llama-index-llms-openai inferedge-moss")
    except Exception as e:
        print(f"An error occurred: {e}")
