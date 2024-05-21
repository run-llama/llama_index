import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
import argparse


def load_vector_database(username, password):
    db_name = "example_db"
    host = "localhost"
    password = password
    port = "5432"
    user = username
    # conn = psycopg2.connect(connection_string)
    conn = psycopg2.connect(
        dbname="postgres",
        host=host,
        password=password,
        port=port,
        user=user,
    )
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")

    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name="llama2_paper",
        embed_dim=384,  # openai embedding dimension
    )
    return vector_store


def load_data(data_path):
    loader = PyMuPDFReader()
    documents = loader.load(file_path=data_path)

    text_parser = SentenceSplitter(
        chunk_size=1024,
        # separator=" ",
    )
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    from llama_index.core.schema import TextNode

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
    return nodes


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


# Transform a list of chat messages into zephyr-specific input
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


def main(args):
    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_path)

    # Use custom LLM in BigDL
    from ipex_llm.llamaindex.llms import IpexLLM

    llm = IpexLLM.from_model_id(
        model_name=args.model_path,
        tokenizer_name=args.tokenizer_path,
        context_window=512,
        max_new_tokens=args.n_predict,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        model_kwargs={},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map=args.device,
    )

    vector_store = load_vector_database(username=args.user, password=args.password)
    nodes = load_data(data_path=args.data)
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    vector_store.add(nodes)

    # query_str = "Can you tell me about the key concepts for safety finetuning"
    query_str = "Explain about the training data for Llama 2"
    query_embedding = embed_model.get_query_embedding(query_str)
    # construct vector store query

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )
    # returns a VectorStoreQueryResult
    query_result = vector_store.query(vector_store_query)
    # print("Retrieval Results: ")
    # print(query_result.nodes[0].get_content())

    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))

    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=1
    )

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    # query_str = "How does Llama 2 perform compared to other open-source models?"
    query_str = args.question
    response = query_engine.query(query_str)

    print("------------RESPONSE GENERATION---------------------")
    print(str(response))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LlamaIndex BigdlLLM Example")
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="the path to transformers model",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default="How does Llama 2 perform compared to other open-source models?",
        help="question you want to ask.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="./data/llama2.pdf",
        help="the data used during retrieval",
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        required=True,
        help="user name in the database postgres",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        required=True,
        help="the password of the user in the database",
    )
    parser.add_argument(
        "-e",
        "--embedding-model-path",
        default="BAAI/bge-small-en",
        help="the path to embedding model path",
    )
    parser.add_argument(
        "-n", "--n-predict", type=int, default=32, help="max number of predict tokens"
    )
    parser.add_argument(
        "-t",
        "--tokenizer-path",
        type=str,
        required=True,
        help="the path to transformers tokenizer",
    )
    parser.add_argument(
        "-x",
        "--device",
        type=str,
        default="xpu",
        help="device to load the model and inference",
    )
    args = parser.parse_args()

    main(args)
