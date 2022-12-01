"""Test embedding functionalities."""

from typing import Any, Tuple
import os
from unittest.mock import patch

from gpt_index.prompts.base import Prompt
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index import GPTTreeIndex
from gpt_index.indices.utils import get_sorted_node_list
from gpt_index.embeddings.utils import (
    get_query_embedding,
    get_text_embedding,
    save_embedding,
    load_embedding,
    get_embedding_similarity,
)
from openai.embeddings_utils import cosine_similarity

TEXT_EMBED_FILE_PATH_TEMPLATE = "data/embeddings/text_embed_{}.txt"
QUERY_STR = "What are the airports in New York City?"
QUERY_EMBED_FILE_PATH = "data/embeddings/query_embed.txt"
NUM_CHUNKS = 103


text_embed_dict = {}
for i in range(NUM_CHUNKS):
    with open(TEXT_EMBED_FILE_PATH_TEMPLATE.format(i), "r") as f:
        embedding = [float(x) for x in f.readline().strip().split(",")]
        text = f.readline().strip()
        text_embed_dict[text] = embedding


def _setup_embeddings() -> None:
    """Load index graph and save embeddings to file."""
    # Load premade index graph
    tree_index = GPTTreeIndex.load_from_disk("data/index.json")
    nodes = get_sorted_node_list(tree_index.index_struct.all_nodes)

    # Create dir if it doesn't exist
    if not os.path.exists("data/embeddings"):
        os.makedirs("data/embeddings")
    # Create embeddings for each chunk and save them to disk
    for i, node in enumerate(nodes):
        file_path = TEXT_EMBED_FILE_PATH_TEMPLATE.format(i)
        node_embedding = get_text_embedding(node.text)
        save_embedding(node_embedding, file_path)
        with open(file_path, "a") as f:
            f.write("\n")
            f.write(node.text.strip().replace("\n", " "))

    # Create embedding for query
    query_embedding = get_query_embedding(QUERY_STR)
    save_embedding(query_embedding, QUERY_EMBED_FILE_PATH)
    with open(QUERY_EMBED_FILE_PATH, "a") as f:
        f.write("\n")
        f.write(QUERY_STR)


def test_embedding_similarity() -> None:
    """Test embedding similarity."""
    # Load pregenerated text embeddings
    text_embeddings = [
        load_embedding(TEXT_EMBED_FILE_PATH_TEMPLATE.format(i))
        for i in range(NUM_CHUNKS)
    ]
    query_embedding = load_embedding(QUERY_EMBED_FILE_PATH)

    similarities = [
        get_embedding_similarity(query_embedding, text_embedding)
        for text_embedding in text_embeddings
    ]

    best_index = similarities.index(max(similarities))
    assert best_index == 85


def _mock_query_text_embedding_similarity(query: str, text: str, mode: str) -> float:
    """Mock get_query_text_embedding_similarity."""
    text_embed = text_embed_dict[text.strip().replace("\n", " ")]
    query_embed = load_embedding(QUERY_EMBED_FILE_PATH)
    return cosine_similarity(query_embed, text_embed)


def _mock_llm_predict(prompt: Prompt, **promp_args: Any) -> Tuple[str, str]:
    formatted_prompt = prompt.format(**promp_args)
    with (open("data/llm_prompt.txt", "r")) as f:
        orig_prompt = f.read()

    if formatted_prompt == orig_prompt:
        response = (
            "The airports in New York City are John F. Kennedy International Airport, "
            + "LaGuardia Airport, and Newark Liberty International Airport."
        )
    else:
        raise ValueError("Prompt not found")
    return response, formatted_prompt


@patch.object(LLMPredictor, "__init__", return_value=None)
@patch.object(LLMPredictor, "predict", side_effect=_mock_llm_predict)
@patch("gpt_index.indices.tree.embedding_query.get_query_text_embedding_similarity")
def test_embedding_query(
    get_query_text_embedding_similarity: Any, _mock_predict: Any, _mock_init: Any
) -> None:
    """Test embedding query."""
    # Load premade index graph
    tree_index = GPTTreeIndex.load_from_disk("data/index.json")

    # Patch utils.get_query_text_embedding_similarity
    get_query_text_embedding_similarity.side_effect = (
        _mock_query_text_embedding_similarity
    )

    response = tree_index.query(QUERY_STR, mode="embedding")

    assert (
        response
        == "The airports in New York City are John F. Kennedy "
        + "International Airport, LaGuardia Airport, and Newark Liberty "
        + "International Airport."
    )


if __name__ == "__main__":
    # setup_embeddings()
    # test_embedding_similarity()
    test_embedding_query()
