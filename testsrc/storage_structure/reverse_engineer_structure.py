import logging
import sys

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext, Settings,
)

def main():
    storage_context = StorageContext.from_defaults()

    documents = SimpleDirectoryReader(input_files=["/tmp/foo.txt"]).load_data()
    documents[0].id_ = "DOC_1"
    splitter = SentenceSplitter(
        chunk_size=15,
        chunk_overlap=0,
        id_func=lambda i, node: f"NODE_{i}",
    )
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter], storage_context=storage_context)

    index.set_index_id("vector_index")
    index.storage_context.persist("./storage/reverse_engineer_structure")

    # Load the index from storage
    storage_context = StorageContext.from_defaults(persist_dir="./storage/reverse_engineer_structure")
    index = load_index_from_storage(storage_context, index_id="vector_index")


if __name__ == '__main__':
    llm = OpenAI(
        api_base="http://localhost:5000/v1",
        api_key="sk-ollama",
        model="gpt-4-turbo",
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.context_window = 8192
    main()

