"""Network Query Engine.

Make sure the app in `contributor.py` is running before trying to run this
script. Run `python contributor.py`.
"""

import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.networks.contributor.query_engine import ContributorClient
from llama_index.networks.network.query_engine import NetworkQueryEngine

client = ContributorClient.from_config_file(env_file=".env.contributor.client")

# build NetworkRAG
llm = OpenAI()
network_query_engine = NetworkQueryEngine.from_args(contributors=[client], llm=llm)

if __name__ == "__main__":
    sync_res = network_query_engine.query("Who is paul")
    print(sync_res)
    print("\n")

    async_res = asyncio.run(network_query_engine.aquery("Who is paul"))
    print(async_res)
