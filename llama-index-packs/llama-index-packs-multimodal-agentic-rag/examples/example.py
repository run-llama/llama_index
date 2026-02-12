import os
# Optional: Set HF mirror for specific regions (can be removed for global users)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import asyncio
from dotenv import load_dotenv
from llama_index.packs.multimodal_agentic_rag import MultimodalAgenticRAGPack

# Load environment variables from .env file
load_dotenv()

async def main():
    # 1. Check API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ Error: DASHSCOPE_API_KEY not found. Please export it or set it in your .env file.")
        return

    print("🚀 [1/3] Initializing Pack...")
    pack = MultimodalAgenticRAGPack(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or "",
        qdrant_url="http://localhost:6333",
        neo4j_url="bolt://localhost:7687",
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"), # Optional: For web search capabilities
        data_dir="./data_test",
        force_recreate=True     # ⚠️ Warning: Setting this to True clears existing DB collections!
    )

    # 2. Prepare Test File
    pdf_path = "test.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ Error: '{pdf_path}' not found.")
        return
    
    # 3. Run Ingestion
    print(f"\n🚀 [2/3] Starting Ingestion: {pdf_path}")
    # Pipeline: Parsing -> Sidecar Gen -> Entity Deduplication -> Qdrant -> Neo4j
    await pack.run_ingestion(pdf_path)

    # 4. Run Retrieval + Generation
    query = "What are the core technologies discussed in this document?"
    print(f"\n❓ [3/3] Querying: {query}")

    response = await pack.run(query)

    # 5. Process Results
    print("\n✅ === Final Response ===")

    if isinstance(response, dict):
        print("🤖 AI Answer:")
        print("-" * 30)
        # Most LlamaPacks return a streaming object for the final response
        async for chunk in response.get("final_response"):
            print(chunk.delta or "", end="", flush=True)

        # 2. Inspecting Visual Metadata (The Sidecar Highlight!)
        print("\n\n📚 Visual Evidence (BBox Metadata):")
        print("-" * 30)
        retrieved_nodes = response.get("retrieved_nodes", [])

        for i, node in enumerate(retrieved_nodes):
            if isinstance(node, dict):
                meta = node.get("metadata", {})
                score = node.get("score", 0.0)
                text = node.get("text", "")
            else:
                meta = node.metadata if hasattr(node, 'metadata') else node.node.metadata
                score = getattr(node, 'score', 0.0)
                text = node.get_content() if hasattr(node, 'get_content') else ""

            if "bbox" in meta:
                print(f"[{i+1}] Page {meta.get('page_label', 'N/A')}: BBox found ✅")
                print(f"    Score: {score:.4f}")
                print(f"    Coordinates: {meta['bbox']}")
            else:
                print(f"[{i+1}] Page {meta.get('page_label', 'N/A')}: No BBox metadata.")
    else:
        # Fallback if response is just a string
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
