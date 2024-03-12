# Required Environment Variables: OPENAI_API_KEY

from llama_index.packs.arize_phoenix_query_engine import ArizePhoenixQueryEnginePack
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from tqdm.auto import tqdm

# load documents and create the pack
documents = SimpleWebPageReader().load_data(
    [
        "https://raw.githubusercontent.com/jerryjliu/llama_index/adb054429f642cc7bbfcb66d4c232e072325eeab/examples/paul_graham_essay/data/paul_graham_essay.txt"
    ]
)
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)
phoenix_pack = ArizePhoenixQueryEnginePack(nodes=nodes)

# run the pack and test queries
queries = [
    "What did Paul Graham do growing up?",
    "When and how did Paul Graham's mother die?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in tqdm(queries):
    print("Query")
    print("=====")
    print(query)
    print()
    response = phoenix_pack.run(query)
    print("Response")
    print("========")
    print(response)
    print()

phoenix_session_url = phoenix_pack.get_modules()["session_url"]
print(f"Open the Phoenix UI to view your trace data: {phoenix_session_url}")
