from typing import List
import math

from llama_index.core.settings import Settings


def lexical_overlap(a: str, b: str) -> float:
    """
    Compute token-level Jaccard similarity between two strings.
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = a_tokens.intersection(b_tokens)
    union = a_tokens.union(b_tokens)

    return len(intersection) / len(union)


async def embedding_distance(a: str, b: str) -> float:
    """
    Compute cosine distance between embeddings of two strings.
    """
    embed_model = Settings.embed_model

    a_vec = await embed_model.aget_text_embedding(a)
    b_vec = await embed_model.aget_text_embedding(b)

    dot = sum(x * y for x, y in zip(a_vec, b_vec))
    norm_a = math.sqrt(sum(x * x for x in a_vec))
    norm_b = math.sqrt(sum(y * y for y in b_vec))

    if norm_a == 0 or norm_b == 0:
        return 1.0

    cosine_sim = dot / (norm_a * norm_b)
    return 1.0 - cosine_sim  # distance


def significant_drift(
    lexical_score: float,
    embedding_dist: float,
    lexical_threshold: float = 0.5,
    embedding_threshold: float = 0.3,
) -> bool:
    """
    Decide whether drift is significant.
    """
    return (
        lexical_score < lexical_threshold
        or embedding_dist > embedding_threshold
    )
