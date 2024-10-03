import os
import logging
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader, StorageContext
from llama_index.graph_stores.memgraph import MemgraphGraphStore


# Step 1: Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"  # Replace with your OpenAI API key

# Step 2: Configure logging
logging.basicConfig(level=logging.INFO)

# Step 3: Configure OpenAI LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.llm = llm
Settings.chunk_size = 512

# Step 4: Write documents to text files (Simulating loading documents from disk)
documents = {
    "doc1.txt": "Python is a popular programming language known for its readability and simplicity. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It is widely used in web development, data science, artificial intelligence, and scientific computing.",
    "doc2.txt": "JavaScript is a high-level programming language primarily used for web development. It was created by Brendan Eich and first appeared in 1995. JavaScript is a core technology of the World Wide Web, alongside HTML and CSS. It enables interactive web pages and is an essential part of web applications. JavaScript is also used in server-side development with environments like Node.js.",
    "doc3.txt": "Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible. It was developed by James Gosling and first released by Sun Microsystems in 1995. Java is widely used for building enterprise-scale applications, mobile applications, and large systems development.",
}

for filename, content in documents.items():
    with open(filename, "w") as file:
        file.write(content)

# Step 5: Load documents
loaded_documents = SimpleDirectoryReader(".").load_data()

# Step 6: Set up Memgraph connection
username = ""  # Enter your Memgraph username (default "")
password = ""  # Enter your Memgraph password (default "")
url = ""  # Specify the connection URL, e.g., 'bolt://localhost:7687'
database = "memgraph"  # Name of the database, default is 'memgraph'

graph_store = MemgraphGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Step 7: Create a Knowledge Graph Index
index = KnowledgeGraphIndex.from_documents(
    loaded_documents,
    storage_context=storage_context,
    max_triplets_per_chunk=3,
)

# Step 8: Query the Knowledge Graph
query_engine = index.as_query_engine(include_text=False, response_mode="tree_summarize")
response = query_engine.query("Tell me about Python and its uses")

print("Query Response:")
print(response)
