from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.ingestion.ray import RayIngestionPipeline, RayTransformComponent
from llama_index.core.schema import Document
import ray

if __name__ == "__main__":
    ray.init()
    documents = [
        Document(text="test 123 " * 100)
    ]  # SimpleDirectoryReader(input_dir="./data/source_files").load_data()

    pipeline = RayIngestionPipeline(
        transformations=[
            RayTransformComponent(SentenceSplitter, chunk_size=1024, chunk_overlap=20),
            RayTransformComponent(OpenAIEmbedding),
        ]
    )

    nodes = pipeline.run(documents=documents)
