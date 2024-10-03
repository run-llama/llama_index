import os
import urllib.request
import nest_asyncio
import logging
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor


# 1. Setup OpenAI API Key (replace this with your actual key)
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"  # Replace with your OpenAI API key

# 2. Create the data directory and download the Paul Graham essay
os.makedirs("data/paul_graham/", exist_ok=True)

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
output_path = "data/paul_graham/paul_graham_essay.txt"
urllib.request.urlretrieve(url, output_path)

# 3. Ensure nest_asyncio is applied
nest_asyncio.apply()

# Step 2: Read the file, replace single quotes, and save the modified content
with open(output_path, "r", encoding="utf-8") as file:
    content = file.read()

# Replace single quotes with escaped single quotes
modified_content = content.replace("'", "\\'")

# Save the modified content back to the same file
with open(output_path, "w", encoding="utf-8") as file:
    file.write(modified_content)

# 4. Load the document data
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# 5. Setup Memgraph connection (ensure Memgraph is running)
username = ""  # Enter your Memgraph username (default "")
password = ""  # Enter your Memgraph password (default "")
url = ""  # Specify the connection URL, e.g., 'bolt://localhost:7687'

graph_store = MemgraphPropertyGraphStore(
    username=username,
    password=password,
    url=url,
)

# 6. Create the Property Graph Index
index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-ada-002"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0),
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

# 7. Querying the graph
retriever = index.as_retriever(include_text=False)

# Example query: "What happened at Interleaf and Viaweb?"
nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

# Output results
print("Query Results:")
for node in nodes:
    print(node.text)

# Alternatively, using a query engine
query_engine = index.as_query_engine(include_text=True)

# Perform a query and print the detailed response
response = query_engine.query("What happened at Interleaf and Viaweb?")
print("\nDetailed Query Response:")
print(str(response))
