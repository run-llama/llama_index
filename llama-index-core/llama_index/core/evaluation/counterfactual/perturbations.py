from typing import List, Sequence
import random


def remove_top_k_contexts(
    contexts: Sequence[str],
    k: int = 1,
) -> List[str]:
    """
    Remove the top-k contexts (assumed to be most salient).

    This simulates evidence removal to test answer dependence.
    """
    if contexts is None:
        return []

    if k <= 0:
        return list(contexts)

    if k >= len(contexts):
        return []

    return list(contexts[k:])


def replace_context_with_decoy(
    contexts: Sequence[str],
    index: int,
    decoy_context: str,
) -> List[str]:
    """
    Replace a specific context with a decoy (semantically similar but incorrect).

    This tests whether the model is robust to spurious grounding.
    """
    if contexts is None:
        return []

    if index < 0 or index >= len(contexts):
        return list(contexts)

    new_contexts = list(contexts)
    new_contexts[index] = decoy_context
    return new_contexts


def random_context_dropout(
    contexts: Sequence[str],
    dropout_prob: float = 0.3,
    seed: int | None = None,
) -> List[str]:
    """
    Randomly drop contexts with a given probability.

    Acts as a control perturbation.
    """
    if contexts is None:
        return []

    if seed is not None:
        random.seed(seed)

    kept = [
        ctx for ctx in contexts if random.random() > dropout_prob
    ]

    # Ensure at least one context remains
    if not kept and contexts:
        kept.append(contexts[0])

    return kept
