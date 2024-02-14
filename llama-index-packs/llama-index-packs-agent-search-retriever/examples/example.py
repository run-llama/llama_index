# Required Environment Variables: SCIPHI_API_KEY

import os
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
AgentSearchRetrieverPack = download_llama_pack(
    "AgentSearchRetrieverPack", "./agent_search_pack"
)

# create the pack
agent_search_pack = AgentSearchRetrieverPack(
    api_key=os.getenv("SCIPHI_API_KEY"),
    similarity_top_k=4,
    search_provider="agent-search",
)

# run the pack
retriever = agent_search_pack.retriever
query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query("Tell me about agent search")
print(response)
