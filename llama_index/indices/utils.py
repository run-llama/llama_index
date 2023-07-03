"""Utilities for GPT indices."""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from llama_index.schema import BaseNode, MetadataMode
from llama_index.utils import globals_helper, truncate_text
from llama_index.vector_stores.types import VectorStoreQueryResult
from langchain.docstore.base import Docstore
from llama_index.embeddings.base import BaseEmbedding

_logger = logging.getLogger(__name__)


def get_sorted_node_list(node_dict: Dict[int, BaseNode]) -> List[BaseNode]:
    """Get sorted node list. Used by tree-strutured indices."""
    sorted_indices = sorted(node_dict.keys())
    return [node_dict[index] for index in sorted_indices]


def get_node_ids_from_doc_ids(
    doc_ids: List[str], docstore: Docstore
) -> List[List[str]]:
    """Get nodes ids from doc ids."""
    node_ids_for_docs: List[List[str]] = []
    for doc_id in doc_ids:
        doc_info = docstore.get_ref_doc_info(doc_id)
        if doc_info is not None and doc_info.node_ids is not None:
            node_ids_for_docs.append(doc_info.node_ids)
        else:
            node_ids_for_docs.append([])
    return node_ids_for_docs


def get_mean_document_embeddings(
    doc_ids: List[str], docstore: Docstore, embed_model: BaseEmbedding
):
    """Get document embeddings."""

    sorted_node_ids_for_docs = get_node_ids_from_doc_ids(doc_ids, docstore)

    # get the embeddings for each node in each retrieved document
    for node_ids in sorted_node_ids_for_docs:
        for node_id in node_ids:
            _node = docstore.get_document(node_id)
            if _node is not None:
                embed_model.queue_text_for_embedding(
                    _node.node_id,
                    _node.get_content(metadata_mode=MetadataMode.EMBED),
                )

    # get the embeddings for each doc by averaging the doc node embeddings
    _, text_embeddings = embed_model.get_queued_text_embeddings()

    idx_offset = 0
    doc_embeddings = []
    for node_ids in sorted_node_ids_for_docs:
        doc_embedding = np.mean(
            text_embeddings[idx_offset : idx_offset + len(node_ids)], axis=0
        )
        doc_embeddings.append(doc_embedding)
        idx_offset += len(node_ids)
    return doc_embeddings


def extract_numbers_given_response(response: str, n: int = 1) -> Optional[List[int]]:
    """Extract number given the GPT-generated response.

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
    """Default format node batch function.

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
) -> Tuple[List[int], Optional[List[float]]]:
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
        answer_num = int(line_tokens[0].split(":")[1].strip())
        if answer_num > num_choices:
            continue
        answer_nums.append(answer_num)
        answer_relevances.append(float(line_tokens[1].split(":")[1].strip()))
    return answer_nums, answer_relevances
