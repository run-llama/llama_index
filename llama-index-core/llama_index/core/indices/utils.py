"""Utilities for GPT indices."""
import logging
import re
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import BaseNode, ImageNode, MetadataMode
from llama_index.core.utils import globals_helper, truncate_text
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from typing import Dict, List, Optional, Sequence, Set, Tuple

_logger = logging.getLogger(__name__)


def get_sorted_node_list(node_dict: Dict[int, BaseNode]) -> List[BaseNode]:
    """Get sorted node list. Used by tree-strutured indices."""
    sorted_indices = sorted(node_dict.keys())
    return [node_dict[index] for index in sorted_indices]


def extract_numbers_given_response(response: str, n: int = 1) -> Optional[List[int]]:
    """
    Extract number given the GPT-generated response.

    Used by tree-structured indices.

    """
    numbers = re.findall(r"\d+", response)
    if len(numbers) == 0:
        return None
    else:
        return numbers[:n]


def expand_tokens_with_subtokens(tokens: Set[str]) -> Set[str]:
    """Get subtokens from a list of tokens., filtering for stopwords."""
    results = set()
    for token in tokens:
        results.add(token)
        sub_tokens = re.findall(r"\w+", token)
        if len(sub_tokens) > 1:
            results.update({w for w in sub_tokens if w not in globals_helper.stopwords})

    return results


def log_vector_store_query_result(
    result: VectorStoreQueryResult, logger: Optional[logging.Logger] = None
) -> None:
    """Log vector store query result."""
    logger = logger or _logger

    assert result.ids is not None
    assert result.nodes is not None
    similarities = (
        result.similarities
        if result.similarities is not None and len(result.similarities) > 0
        else [1.0 for _ in result.ids]
    )

    fmt_txts = []
    for node_idx, node_similarity, node in zip(result.ids, similarities, result.nodes):
        fmt_txt = f"> [Node {node_idx}] [Similarity score: \
            {float(node_similarity):.6}] {truncate_text(node.get_content(), 100)}"
        fmt_txts.append(fmt_txt)
    top_k_node_text = "\n".join(fmt_txts)
    logger.debug(f"> Top {len(result.nodes)} nodes:\n{top_k_node_text}")


def default_format_node_batch_fn(
    summary_nodes: List[BaseNode],
) -> str:
    """
    Default format node batch function.

    Assign each summary node a number, and format the batch of nodes.

    """
    fmt_node_txts = []
    for idx in range(len(summary_nodes)):
        number = idx + 1
        fmt_node_txts.append(
            f"Document {number}:\n"
            f"{summary_nodes[idx].get_content(metadata_mode=MetadataMode.LLM)}"
        )
    return "\n\n".join(fmt_node_txts)


def default_parse_choice_select_answer_fn(
    answer: str, num_choices: int, raise_error: bool = False
) -> Tuple[List[int], List[float]]:
    """Default parse choice select answer function."""
    answer_lines = answer.split("\n")
    answer_nums = []
    answer_relevances = []
    for answer_line in answer_lines:
        line_tokens = answer_line.split(",")
        if len(line_tokens) != 2:
            if not raise_error:
                continue
            else:
                raise ValueError(
                    f"Invalid answer line: {answer_line}. "
                    "Answer line must be of the form: "
                    "answer_num: <int>, answer_relevance: <float>"
                )
        try:
            answer_num = int(line_tokens[0].split(":")[1].strip())
        except (IndexError, ValueError) as e:
            if not raise_error:
                continue
            else:
                raise ValueError(
                    f"Invalid answer line: {answer_line}. "
                    "Answer line must be of the form: "
                    "answer_num: <int>, answer_relevance: <float>"
                )
        if answer_num > num_choices:
            continue
        answer_nums.append(answer_num)
        # extract just the first digits after the colon.
        try:
            _answer_relevance = re.findall(
                r"\d+", line_tokens[1].split(":")[1].strip()
            )[0]
            answer_relevances.append(float(_answer_relevance))
        except (IndexError, ValueError) as e:
            if not raise_error:
                continue
            else:
                raise ValueError(
                    f"Invalid answer line: {answer_line}. "
                    "Answer line must be of the form: "
                    "answer_num: <int>, answer_relevance: <float>"
                )
    return answer_nums, answer_relevances


def embed_nodes(
    nodes: Sequence[BaseNode], embed_model: BaseEmbedding, show_progress: bool = False
) -> Dict[str, List[float]]:
    """
    Get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode]): The nodes to embed.
        embed_model (BaseEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        Dict[str, List[float]]: A map from node id to embedding.

    """
    id_to_embed_map: Dict[str, List[float]] = {}

    texts_to_embed = []
    ids_to_embed = []
    for node in nodes:
        if node.embedding is None:
            ids_to_embed.append(node.node_id)
            texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))
        else:
            id_to_embed_map[node.node_id] = node.embedding

    new_embeddings = embed_model.get_text_embedding_batch(
        texts_to_embed, show_progress=show_progress
    )

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = text_embedding

    return id_to_embed_map


def embed_image_nodes(
    nodes: Sequence[ImageNode],
    embed_model: MultiModalEmbedding,
    show_progress: bool = False,
) -> Dict[str, List[float]]:
    """
    Get image embeddings of the given nodes, run image embedding model if necessary.

    Args:
        nodes (Sequence[ImageNode]): The nodes to embed.
        embed_model (MultiModalEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        Dict[str, List[float]]: A map from node id to embedding.

    """
    id_to_embed_map: Dict[str, List[float]] = {}

    images_to_embed = []
    ids_to_embed = []
    for node in nodes:
        if node.embedding is None:
            ids_to_embed.append(node.node_id)
            images_to_embed.append(node.resolve_image())
        else:
            id_to_embed_map[node.node_id] = node.embedding

    new_embeddings = embed_model.get_image_embedding_batch(
        images_to_embed, show_progress=show_progress
    )

    for new_id, img_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = img_embedding

    return id_to_embed_map


async def async_embed_nodes(
    nodes: Sequence[BaseNode], embed_model: BaseEmbedding, show_progress: bool = False
) -> Dict[str, List[float]]:
    """
    Async get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode]): The nodes to embed.
        embed_model (BaseEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        Dict[str, List[float]]: A map from node id to embedding.

    """
    id_to_embed_map: Dict[str, List[float]] = {}

    texts_to_embed = []
    ids_to_embed = []
    for node in nodes:
        if node.embedding is None:
            ids_to_embed.append(node.node_id)
            texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))
        else:
            id_to_embed_map[node.node_id] = node.embedding

    new_embeddings = await embed_model.aget_text_embedding_batch(
        texts_to_embed, show_progress=show_progress
    )

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = text_embedding

    return id_to_embed_map


async def async_embed_image_nodes(
    nodes: Sequence[ImageNode],
    embed_model: MultiModalEmbedding,
    show_progress: bool = False,
) -> Dict[str, List[float]]:
    """
    Get image embeddings of the given nodes, run image embedding model if necessary.

    Args:
        nodes (Sequence[ImageNode]): The nodes to embed.
        embed_model (MultiModalEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        Dict[str, List[float]]: A map from node id to embedding.

    """
    id_to_embed_map: Dict[str, List[float]] = {}

    images_to_embed = []
    ids_to_embed = []
    for node in nodes:
        if node.embedding is None:
            ids_to_embed.append(node.node_id)
            images_to_embed.append(node.resolve_image())
        else:
            id_to_embed_map[node.node_id] = node.embedding

    new_embeddings = await embed_model.aget_image_embedding_batch(
        images_to_embed, show_progress=show_progress
    )

    for new_id, img_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = img_embedding

    return id_to_embed_map
