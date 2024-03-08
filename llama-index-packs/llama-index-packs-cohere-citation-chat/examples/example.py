from llama_index.packs.cohere_citation_chat import CohereCitationChatEnginePack
from llama_index.readers.web import SimpleWebPageReader

# load documents and create the pack
documents = SimpleWebPageReader().load_data(
    [
        "https://raw.githubusercontent.com/jerryjliu/llama_index/adb054429f642cc7bbfcb66d4c232e072325eeab/examples/paul_graham_essay/data/paul_graham_essay.txt"
    ]
)
cohere_citation_chat_pack = CohereCitationChatEnginePack(
    documents=documents, cohere_api_key="your-api-key"
)
chat_engine = cohere_citation_chat_pack.run()
# run the pack and test queries
queries = [
    "What did Paul Graham do growing up?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in queries:
    print("Query ")
    print("=====")
    print(query)
    print("Chat")
    response = chat_engine.chat(query)
    print("Chat Response")
    print("========")
    print(response)
    print(f"Citations: {response.citations}")
    print(f"Documents: {response.documents}")
    print("Stream Chat")
    response = chat_engine.stream_chat(query)
    print("Stream Chat Response")
    print("========")
    response.print_response_stream()
    print(f"Citations: {response.citations}")
    print(f"Documents: {response.documents}")
