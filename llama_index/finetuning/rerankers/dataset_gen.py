import json
import random
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple

from llama_index.finetuning import EmbeddingQAFinetuneDataset


@dataclass
class CohereRerankerFinetuneDataset:
    """Class for keeping track of CohereAI Reranker finetuning training/ validation Dataset."""

    query: str
    relevant_passages: list
    hard_negatives: list

    def to_jsonl(self) -> str:
        """Convert the dataclass instance to a JSONL string."""
        return json.dumps(asdict(self)) + "\n"


def generate_embeddings(embed_model: Any, text: str) -> List[float]:
    # Generate embeddings for a list of texts
    return embed_model.get_text_embedding(text)


def generate_hard_negatives(
    queries: List[str],
    relevant_contexts: List[str],
    embed_model: Optional[Any],
    num_negatives: int = 5,
    method: str = "random",
    top_k_dissimilar: int = 100,
) -> Any:
    from sklearn.metrics.pairwise import cosine_similarity

    hard_negatives = []

    if method == "cosine_similarity":
        query_embeddings = [
            generate_embeddings(embed_model, query) for query in queries
        ]
        relevant_contexts_embeddings = [
            generate_embeddings(embed_model, context) for context in relevant_contexts
        ]
        # Calculate cosine similarity between queries and context embeddings
        similarity_matrix = cosine_similarity(
            query_embeddings, relevant_contexts_embeddings
        )
    for i, _ in enumerate(queries):
        if method == "random":
            # Exclude the correct context
            potential_negatives = relevant_contexts[:i] + relevant_contexts[i + 1 :]
            # Randomly select hard negatives
            hard_negatives.append(
                random.sample(
                    potential_negatives, min(num_negatives, len(potential_negatives))
                )
            )

        elif method == "cosine_similarity":
            # Exclude the similarity score for the correct context
            potential_negatives_scores = list(enumerate(similarity_matrix[i]))
            potential_negatives_scores.pop(i)  # remove the correct context score
            # Sort based on similarity scores
            potential_negatives_scores.sort(key=lambda x: x[1], reverse=False)
            # Take the top 'top_k_dissimilar' least similar contexts
            least_similar_contexts = [
                relevant_contexts[idx]
                for idx, _ in potential_negatives_scores[:top_k_dissimilar]
            ]

            # Randomly select hard negatives from the least similar contexts
            hard_negatives.append(
                random.sample(
                    least_similar_contexts,
                    min(num_negatives, len(least_similar_contexts)),
                )
            )
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
    embed_model: Optional[Any],
    num_negatives: int = 5,
    top_k_dissimilar: int = 100,
    hard_negatives_required: bool = False,
    hard_negatives_gen_method: str = "random",
    finetune_dataset_file_name: str = "train.jsonl",
) -> Any:
    queries, relevant_contexts = get_query_context_lists(query_context_pairs)

    if hard_negatives_required:
        hard_negatives = generate_hard_negatives(
            queries,
            relevant_contexts,
            embed_model,
            num_negatives,
            hard_negatives_gen_method,
            top_k_dissimilar,
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
