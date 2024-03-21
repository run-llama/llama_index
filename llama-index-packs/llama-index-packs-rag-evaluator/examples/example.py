# Required Environment Variables: OPENAI_API_KEY

from llama_index.core import VectorStoreIndex
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.packs.rag_evaluator import RagEvaluatorPack

# download a LabelledRagDataset from llama-hub
rag_dataset, documents = download_llama_dataset(
    "PaulGrahamEssayDataset", "./paul_graham"
)

# build a basic RAG pipeline off of the source documents
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

# construction requires a query_engine, a rag_dataset, and optionally a judge_llm
rag_evaluator_pack = RagEvaluatorPack(
    query_engine=query_engine, rag_dataset=rag_dataset
)

# PERFORM EVALUATION
benchmark_df = rag_evaluator_pack.run()  # async arun() also supported
print(benchmark_df)
