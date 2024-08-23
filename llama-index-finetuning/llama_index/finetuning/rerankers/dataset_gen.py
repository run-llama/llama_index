import random
from typing import Any, List, Optional, Tuple

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.finetuning import EmbeddingQAFinetuneDataset


class CohereRerankerFinetuneDataset(BaseModel):
    """Class for keeping track of CohereAI Reranker finetuning training/validation Dataset."""

    query: str
    relevant_passages: List[str]
    hard_negatives: Any

    def to_jsonl(self) -> str:
        """Convert the BaseModel instance to a JSONL string."""
        return self.model_dump_json() + "\n"


def generate_embeddings(embed_model: Any, text: str) -> List[float]:
    # Generate embeddings for a list of texts
    return embed_model.get_text_embedding(text)


def generate_hard_negatives(
    queries: List[str],
    relevant_contexts: List[str],
    embed_model: Optional[Any],
    num_negatives: int = 5,
    method: str = "random",
) -> Any:
    hard_negatives = []

    if method == "cosine_similarity":
        query_embeddings = [
            generate_embeddings(embed_model, query) for query in queries
        ]
        relevant_contexts_embeddings = [
            generate_embeddings(embed_model, context) for context in relevant_contexts
        ]

    for query_index, _ in enumerate(queries):
        if method == "random":
            # Exclude the correct context
            potential_negatives = (
                relevant_contexts[:query_index] + relevant_contexts[query_index + 1 :]
            )
            # Randomly select hard negatives
            hard_negatives.append(
                random.sample(
                    potential_negatives, min(num_negatives, len(potential_negatives))
                )
            )

        elif method == "cosine_similarity":
            query_embedding = query_embeddings[query_index]
            # Use get_top_k_embeddings to select num_negatives closest but not correct contexts
            _, relevant_contexts_indices = get_top_k_embeddings(
                query_embedding,
                relevant_contexts_embeddings,
            )

            # Filter out the correct context to only include hard negatives
            hard_negative_indices = [
                idx for idx in relevant_contexts_indices if idx != query_index
            ][:num_negatives]

            # Map indices to actual contexts to get the hard negatives
            hard_negatives_for_query = [
                relevant_contexts[idx] for idx in hard_negative_indices
            ]

            hard_negatives.append(hard_negatives_for_query)
    return hard_negatives


def get_query_context_lists(
    query_context_pairs: EmbeddingQAFinetuneDataset,
) -> Tuple[List[str], List[str]]:
    queries = []
    relevant_contexts = []

    # 'query_context_pairs' is an object with 'queries', 'corpus', and 'relevant_docs' attributes
    for query_id, query in query_context_pairs.queries.items():
        # Get the first relevant document ID for the current query
        relevant_doc_id = query_context_pairs.relevant_docs[query_id][0]
        # Get the relevant context using the relevant document ID
        relevant_context = query_context_pairs.corpus[relevant_doc_id]
        # Append the query and the relevant context to their respective lists
        queries.append(query)
        relevant_contexts.append(relevant_context)

    return queries, relevant_contexts


def generate_cohere_reranker_finetuning_dataset(
    query_context_pairs: EmbeddingQAFinetuneDataset,
    num_negatives: int = 0,
    top_k_dissimilar: int = 100,
    hard_negatives_gen_method: str = "random",
    finetune_dataset_file_name: str = "train.jsonl",
    embed_model: Optional[Any] = None,
) -> Any:
    queries, relevant_contexts = get_query_context_lists(query_context_pairs)

    if num_negatives:
        hard_negatives = generate_hard_negatives(
            queries,
            relevant_contexts,
            embed_model,
            num_negatives,
            hard_negatives_gen_method,
        )
    else:
        hard_negatives = [[] for _ in queries]
    # Open the file in write mode
    with open(finetune_dataset_file_name, "w") as outfile:
        # Iterate over the lists simultaneously using zip
        for query, context, hard_negative in zip(
            queries, relevant_contexts, hard_negatives
        ):
            # Instantiate a CohereRerankerFinetuneDataset object for the current entry
            entry = CohereRerankerFinetuneDataset(
                query=query, relevant_passages=[context], hard_negatives=hard_negative
            )
            # Write the JSONL string to the file
            outfile.write(entry.to_jsonl())
